use std::time::Instant;
use anyhow::{Result, bail, Context};
use rand::seq::SliceRandom;
use std::{fs, collections::HashMap};
use tch::{nn, nn::ModuleT, nn::OptimizerConfig, Device, Kind, Tensor};
use image::{imageops, DynamicImage};
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};

// =============== HYPERPARAM & PATH ===============
const IMG_SIZE: u32 = 224;
const EPOCHS: i64 = 15;
const BATCH_SIZE: usize = 8;
const LEARNING_RATE: f64 = 5e-5;

// MobileNetV2 width multiplier
const WIDTH_MULT: f64 = 1.0;

// Data & model
const DATA_DIR: &str = "../Data";
const MODEL_DIR: &str = "../models";
const MODEL_PATH: &str = "../models/best_mobilenet_v2.safetensors";

// Pretrain & freeze
const LOAD_PRETRAINED: bool = true;
const FREEZE_BASE: bool = true;
// SET ke file yang kamu punya (boleh file asli TIMM atau hasil remap / schema blocks.i.j.*)
const PRETRAIN_PATH: &str = "../weights/mobilenet_v2_1_0_imagenet.safetensors";
const PRETRAIN_NUM_CLASSES: i64 = 1000;

// =============== UTIL ===============
fn round_channels(c: i64, wm: f64) -> i64 {
    let v = (c as f64 * wm).round() as i64;
    v.max(8)
}

fn normalize_tensor(mut tensor: Tensor) -> Tensor {
    tensor = tensor / 255.0;
    (tensor - 0.5) / 0.5
}

fn augment_image(img: DynamicImage, is_training: bool) -> DynamicImage {
    if !is_training { return img; }
    let mut img = img;

    if rand::random::<bool>() { img = img.flipv(); }
    if rand::random::<bool>() { img = img.fliph(); }

    let rotations = [0, 90, 180, 270];
    let angle = *rotations.choose(&mut rand::thread_rng()).unwrap();
    img = match angle {
        0 => img,
        90 => imageops::rotate90(&img).into(),
        180 => imageops::rotate180(&img).into(),
        270 => imageops::rotate270(&img).into(),
        _ => img,
    };

    let brightness: f32 = rand::random::<f32>() * 0.2 - 0.1;
    let contrast: f32 = rand::random::<f32>() * 0.2 + 0.9;
    img.adjust_contrast(contrast).brighten((brightness * 255.0) as i32)
}

// =============== DATA LOADING ===============
fn list_classes(base: &str) -> Result<Vec<String>> {
    let mut classes: Vec<String> = fs::read_dir(format!("{}/train", base))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .filter_map(|p| p.file_name().and_then(|s| s.to_str()).map(|s| s.to_string()))
        .collect();
    classes.sort();
    Ok(classes)
}

fn load_split_dataset_with_classes(
    base: &str,
    split: &str,
    classes: &[String],
    is_training: bool
) -> Result<(Vec<Tensor>, Vec<i64>)> {
    let mut all_images = Vec::new();
    let mut all_labels = Vec::new();

    let split_path = format!("{}/{}", base, split);
    if !std::path::Path::new(&split_path).exists() {
        println!("⚠ Split '{}' not found at: {}", split, split_path);
        return Ok((all_images, all_labels));
    }

    let index_map: HashMap<&str, i64> = classes
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i as i64))
        .collect();

    println!("\n Loading {} dataset...", split.to_uppercase());
    let mut class_entries: Vec<_> = fs::read_dir(&split_path)?.filter_map(|e| e.ok()).collect();
    class_entries.sort_by_key(|e| e.file_name());

    for entry in class_entries {
        let class_dir = entry.path();
        if !class_dir.is_dir() { continue; }

        let class_name = match class_dir.file_name().and_then(|s| s.to_str()) {
            Some(s) => s,
            None => continue,
        };

        let Some(&class_idx) = index_map.get(class_name) else {
            eprintln!("  ⚠ Kelas '{}' tidak ada di daftar global. Lewati.", class_name);
            continue;
        };

        let mut img_paths: Vec<_> = fs::read_dir(&class_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_file())
            .collect();
        img_paths.sort();

        println!("  Class {}: {} ({} images)", class_idx, class_name, img_paths.len());

        let results: Vec<_> = img_paths.par_iter().filter_map(|img_path| {
            match image::open(img_path) {
                Ok(img) => {
                    let img = augment_image(img, is_training);
                    let img = img.resize_exact(IMG_SIZE, IMG_SIZE, imageops::FilterType::Triangle);
                    let img_rgb = img.to_rgb8();
                    let raw = img_rgb.into_raw();
                    let tensor = Tensor::from_slice(&raw)
                        .reshape(&[IMG_SIZE as i64, IMG_SIZE as i64, 3])
                        .to_kind(Kind::Uint8);
                    let tensor = normalize_tensor(tensor.permute(&[2, 0, 1]).to_kind(Kind::Float));
                    Some((tensor, class_idx))
                }
                Err(e) => {
                    eprintln!("  ⚠ Failed to load {:?}: {}", img_path, e);
                    None
                }
            }
        }).collect();

        for (tensor, label) in results {
            all_images.push(tensor);
            all_labels.push(label);
        }
    }

    println!("✓ Loaded {} images\n", all_images.len());
    Ok((all_images, all_labels))
}

// =============== MODEL ===============
#[derive(Debug)]
struct InvertedResidual {
    conv_expand: Option<nn::Conv2D>,
    bn_expand:   Option<nn::BatchNorm>,
    conv_dw:     nn::Conv2D,
    bn_dw:       nn::BatchNorm,
    conv_project: nn::Conv2D,
    bn_project:  nn::BatchNorm,
    use_res:     bool,
}

impl InvertedResidual {
    fn new(vs: &nn::Path, in_c: i64, out_c: i64, stride: i64, expand_ratio: i64) -> Self {
        let hidden_dim = in_c * expand_ratio;

        let (conv_expand, bn_expand) = if expand_ratio != 1 {
            (
                nn::conv2d(&vs.sub("expand"), in_c, hidden_dim, 1, nn::ConvConfig { bias: false, ..Default::default() }).into(),
                nn::batch_norm2d(&vs.sub("expand").sub("bn"), hidden_dim, Default::default()).into(),
            )
        } else { (None, None) };

        let conv_dw = nn::conv2d(
            &vs.sub("dw"), hidden_dim, hidden_dim, 3,
            nn::ConvConfig { stride, padding: 1, groups: hidden_dim, bias: false, ..Default::default() }
        );
        let bn_dw = nn::batch_norm2d(&vs.sub("dw").sub("bn"), hidden_dim, Default::default());

        let conv_project = nn::conv2d(&vs.sub("project"), hidden_dim, out_c, 1, nn::ConvConfig { bias: false, ..Default::default() });
        let bn_project   = nn::batch_norm2d(&vs.sub("project").sub("bn"), out_c, Default::default());

        let use_res = stride == 1 && in_c == out_c;

        Self { conv_expand, bn_expand, conv_dw, bn_dw, conv_project, bn_project, use_res }
    }
}

impl nn::ModuleT for InvertedResidual {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let relu6 = |x: &Tensor| x.clamp(0.0, 6.0);

        let mut x = xs.shallow_clone();
        if let (Some(conv_e), Some(bn_e)) = (&self.conv_expand, &self.bn_expand) {
            x = conv_e.forward_t(&x, train);
            x = bn_e.forward_t(&x, train);
            x = relu6(&x);
        }
        x = self.conv_dw.forward_t(&x, train);
        x = self.bn_dw.forward_t(&x, train);
        x = relu6(&x);

        x = self.conv_project.forward_t(&x, train);
        x = self.bn_project.forward_t(&x, train);

        if self.use_res { xs + x } else { x }
    }
}

// base (features) & head (classifier)
fn mobilenet_v2_base(vs: &nn::Path, alpha: f64) -> (nn::SequentialT, i64) {
    // t, c, n, s
    let cfg: &[(i64, i64, i64, i64)] = &[
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ];
    let relu6 = |x: &Tensor| x.clamp(0.0, 6.0);

    let mut seq = nn::seq_t();

    // stem
    let first = round_channels(32, alpha);
    let stem_vs = vs.sub("stem");
    seq = seq
        .add(nn::conv2d(&stem_vs, 3, first, 3, nn::ConvConfig { stride: 2, padding: 1, bias: false, ..Default::default() }))
        .add(nn::batch_norm2d(&stem_vs.sub("bn"), first, Default::default()))
        .add_fn(move |x| relu6(x));

    // bottlenecks
    let mut in_c = first;
    for (i, &(t, c, n, s)) in cfg.iter().enumerate() {
        let out_c = round_channels(c, alpha);
        for j in 0..n {
            let stride = if j == 0 { s } else { 1 };
            let block_vs = vs.sub(&format!("ir_{}_{}", i, j));
            let block = InvertedResidual::new(&block_vs, in_c, out_c, stride, t);
            seq = seq.add(block);
            in_c = out_c;
        }
    }

    // last conv
    let last = round_channels(1280, alpha);
    let last_vs = vs.sub("last");
    seq = seq
        .add(nn::conv2d(&last_vs, in_c, last, 1, nn::ConvConfig { bias: false, ..Default::default() }))
        .add(nn::batch_norm2d(&last_vs.sub("bn"), last, Default::default()))
        .add_fn(move |x| relu6(x))
        .add_fn(|x| x.adaptive_avg_pool2d(&[1, 1]))
        .add_fn(move |x| x.view([-1, last]));

    (seq, last)
}

fn mobilenet_v2_head(vs: &nn::Path, in_features: i64, num_classes: i64) -> nn::SequentialT {
    nn::seq_t()
        .add_fn_t(|x, train| x.dropout(0.2, train))
        .add(nn::linear(vs, in_features, num_classes, Default::default()))
}

#[derive(Debug)]
struct MobileNetV2Split {
    base: nn::SequentialT,
    head: nn::SequentialT,
}

impl nn::ModuleT for MobileNetV2Split {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let f = self.base.forward_t(xs, train);
        self.head.forward_t(&f, train)
    }
}

// =============== SAFE TENSORS READER (Rust) ===============
use safetensors::{SafeTensors, tensor::Dtype};
use bytemuck::cast_slice;
use half::f16;

fn load_safetensors_to_map(path: &str) -> Result<HashMap<String, Tensor>> {
    let bytes = fs::read(path).with_context(|| format!("read file {}", path))?;
    let st = SafeTensors::deserialize(&bytes).with_context(|| "deserialize safetensors")?;
    let mut map: HashMap<String, Tensor> = HashMap::new();

    for name in st.names() {
        let tv = st.tensor(name).with_context(|| format!("get tensor {}", name))?;
        let shape_i64: Vec<i64> = tv.shape().iter().map(|&d| d as i64).collect();
        let data = tv.data();

        let t = match tv.dtype() {
            Dtype::F32 => {
                let slice: &[f32] = cast_slice(data);
                Tensor::from_slice(slice).reshape(&shape_i64)
            }
            Dtype::F16 => {
                // f16 (u16 bits) -> f32 menggunakan crate `half`
                let slice_u16: &[u16] = cast_slice(data);
                let vec_f32: Vec<f32> = slice_u16.iter()
                    .map(|&h| f16::from_bits(h).to_f32())
                    .collect();
                Tensor::from_slice(&vec_f32).reshape(&shape_i64)
            }
            Dtype::I64 => {
                let slice: &[i64] = cast_slice(data);
                Tensor::from_slice(slice).reshape(&shape_i64).to_kind(Kind::Int64)
            }
            Dtype::I32 => {
                let slice: &[i32] = cast_slice(data);
                // Di tch: pakai Kind::Int untuk int32
                Tensor::from_slice(slice).reshape(&shape_i64).to_kind(Kind::Int)
            }
            other => {
                eprintln!("Skipping tensor {} with unsupported dtype {:?}", name, other);
                continue;
            }
        };

        map.insert(name.to_string(), t);
    }
    Ok(map)
}

// =============== MAPPING HELPERS ===============
fn cfg_spec() -> Vec<(i64,i64,i64,i64)> {
    vec![
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]
}

fn block_idx_to_ij(mut k: usize, cfg: &[(i64,i64,i64,i64)]) -> (usize, usize) {
    for (i, &(_t,_c,n,_s)) in cfg.iter().enumerate() {
        if k < n as usize { return (i, k); }
        k -= n as usize;
    }
    (0,0)
}

fn map_timm_key_to_base(key: &str, cfg: &[(i64,i64,i64,i64)]) -> Option<String> {
    if key == "conv_stem.weight" { return Some("base.stem.weight".into()); }
    if let Some(rest) = key.strip_prefix("bn1.") { return Some(format!("base.stem.bn.{rest}")); }
    if key == "conv_head.weight" { return Some("base.last.weight".into()); }
    if let Some(rest) = key.strip_prefix("bn2.") { return Some(format!("base.last.bn.{rest}")); }
    if let Some(rem) = key.strip_prefix("blocks.") {
        let mut it = rem.splitn(3, '.');
        let k_str = it.next().unwrap_or("");
        let _conv = it.next().unwrap_or("");
        let tail = it.next().unwrap_or("");
        if _conv != "conv" { return None; }
        let k: usize = k_str.parse().ok()?;
        let (i,j) = block_idx_to_ij(k, cfg);
        if let Some(sfx) = tail.strip_prefix("pw.") { return Some(format!("base.ir_{}_{}.expand.{}", i,j,sfx)); }
        if let Some(sfx) = tail.strip_prefix("pw_bn.") { return Some(format!("base.ir_{}_{}.expand.bn.{}", i,j,sfx)); }
        if let Some(sfx) = tail.strip_prefix("dw.") { return Some(format!("base.ir_{}_{}.dw.{}", i,j,sfx)); }
        if let Some(sfx) = tail.strip_prefix("dw_bn.") { return Some(format!("base.ir_{}_{}.dw.bn.{}", i,j,sfx)); }
        if let Some(sfx) = tail.strip_prefix("pw_linear.") { return Some(format!("base.ir_{}_{}.project.{}", i,j,sfx)); }
        if let Some(sfx) = tail.strip_prefix("pw_linear_bn.") { return Some(format!("base.ir_{}_{}.project.bn.{}", i,j,sfx)); }
    }
    None
}

// compute per-block (i,j) in/out/hidden to help TorchVision-ish mapping
#[derive(Clone, Copy)]
struct BlockShape { in_c: i64, out_c: i64, t: i64, hidden: i64 }
fn compute_block_shapes(alpha: f64) -> Vec<Vec<BlockShape>> {
    let cfg = cfg_spec();
    let mut out: Vec<Vec<BlockShape>> = Vec::new();
    let first = round_channels(32, alpha);
    let mut in_c = first;
    for & (t,c,n,_s) in cfg.iter() {
        let out_c = round_channels(c, alpha);
        let mut row = Vec::new();
        for _ in 0..n {
            let hidden = in_c * t;
            row.push(BlockShape { in_c, out_c, t, hidden });
            in_c = out_c;
        }
        out.push(row);
    }
    out
}

// TorchVision-ish: blocks.i.j.{conv_dw,conv_pw,conv_pw_1,bn1,bn2,bn3}
fn map_blocks_ij_to_base(key: &str, shapes: &Vec<Vec<BlockShape>>, tensors: &HashMap<String, Tensor>) -> Option<String> {
    // match "blocks.i.j.xxx"
    let parts: Vec<&str> = key.split('.').collect();
    if parts.len() < 4 || parts[0] != "blocks" { return None; }
    let i: usize = parts[1].parse().ok()?;
    let j: usize = parts[2].parse().ok()?;
    let tail = parts[3..].join(".");

    let b = shapes.get(i)?.get(j)?.clone();

    // convs
    if tail == "conv_dw.weight" { return Some(format!("base.ir_{}_{}.dw.weight", i, j)); }
    if tail == "conv_pw.weight" {
        // disambiguate: if t==1, conv_pw == project; if t>1 bisa expand atau project
        let w = tensors.get(key)?;
        let out_ch = w.size()[0];
        if b.t == 1 { return Some(format!("base.ir_{}_{}.project.weight", i, j)); }
        // choose by out channels
        if out_ch == b.hidden { return Some(format!("base.ir_{}_{}.expand.weight", i, j)); }
        else { return Some(format!("base.ir_{}_{}.project.weight", i, j)); }
    }
    if tail == "conv_pw_1.weight" {
        // definitely project (second pw)
        return Some(format!("base.ir_{}_{}.project.weight", i, j));
    }

    // bn
    // heuristic order:
    //  t==1: bn1 -> dw.bn ; bn2 -> project.bn
    //  t>1 : bn1 -> expand.bn ; bn2 -> dw.bn ; bn3 -> project.bn
    let map_bn = |which: &str, suffix: &str| -> Option<String> {
        if b.t == 1 {
            match which {
                "bn1" => Some(format!("base.ir_{}_{}.dw.bn.{}", i,j,suffix)),
                "bn2" => Some(format!("base.ir_{}_{}.project.bn.{}", i,j,suffix)),
                _ => None
            }
        } else {
            match which {
                "bn1" => Some(format!("base.ir_{}_{}.expand.bn.{}", i,j,suffix)),
                "bn2" => Some(format!("base.ir_{}_{}.dw.bn.{}", i,j,suffix)),
                "bn3" => Some(format!("base.ir_{}_{}.project.bn.{}", i,j,suffix)),
                _ => None
            }
        }
    };

    for bn in &["bn1","bn2","bn3"] {
        let prefix = format!("{}.", bn);
        if let Some(rest) = tail.strip_prefix(&prefix) {
            // ignore num_batches_tracked (PyTorch BN counter, bukan parameter)
            if rest == "num_batches_tracked" { return None; }
            return map_bn(bn, rest);
        }
    }

    None
}

// copy util: try copy if exists & shape match
fn try_copy(dst: &mut std::collections::HashMap<String, Tensor>, name: &str, src: &Tensor) -> bool {
    if let Some(d) = dst.get_mut(name) {
        if d.size() == src.size() {
            // penting: jangan track grad saat inject weight
            tch::no_grad(|| {
                d.copy_(src);
            });
            return true;
        }
    }
    false
}


// MASTER loader: baca safetensors (apa pun), map → base.*, copy partial
fn load_pretrained_base_only(
    _device: Device,
    target_vs: &mut nn::VarStore,
    alpha: f64,
    pretrain_path: &str,
) -> Result<usize> {
    let src = load_safetensors_to_map(pretrain_path)?;
    let cfg = cfg_spec();
    let shapes = compute_block_shapes(alpha);

    // collect target vars (mutable)
    let mut tgt_vars = target_vs.variables();

    // detect schema & map all keys to base.* candidates
    let has_base = src.keys().any(|k| k.starts_with("base."));
    let has_timm = src.contains_key("conv_stem.weight") || src.keys().any(|k| k.starts_with("blocks.") && k.contains(".conv."));
    let has_bij  = src.keys().any(|k| k.starts_with("blocks.") && k.matches('.').count() >= 3 && !k.contains(".conv."));

    let mut copied = 0usize;

    for (k, t) in src.iter() {
        let dst = if has_base {
            // already base.*, pass through
            if k.starts_with("base.") { Some(k.clone()) } else { None }
        } else if has_timm {
            map_timm_key_to_base(k, &cfg)
        } else if has_bij {
            map_blocks_ij_to_base(k, &shapes, &src)
        } else {
            None
        };

        if let Some(dst_name) = dst {
            if try_copy(&mut tgt_vars, &dst_name, t) {
                copied += 1;
            }
        }
    }

    if copied == 0 {
        bail!("no tensors copied from '{}'", pretrain_path);
    }
    Ok(copied)
}

// =============== TRAIN/EVAL ===============
fn batch_from_vec(images: &[Tensor], labels: &[i64], idxs: &[usize], device: Device) -> (Tensor, Tensor) {
    let xs: Vec<Tensor> = idxs.iter().map(|&i| images[i].shallow_clone()).collect();
    let ys: Vec<i64>    = idxs.iter().map(|&i| labels[i]).collect();
    (Tensor::stack(&xs, 0).to(device), Tensor::from_slice(&ys).to(device))
}

fn eval_stream(model: &impl nn::ModuleT, images: &[Tensor], labels: &[i64], device: Device, batch_size: usize) -> (f64, f64) {
    let mut tot_loss = 0.0;
    let mut tot_correct = 0i64;
    let mut tot_seen = 0i64;

    for chunk in (0..images.len()).collect::<Vec<_>>().chunks(batch_size) {
        let (bxs, bys) = batch_from_vec(images, labels, chunk, device);
        let logits = model.forward_t(&bxs, false);
        tot_loss += logits.cross_entropy_for_logits(&bys).double_value(&[]);
        let preds = logits.argmax(-1, false);
        tot_correct += preds.eq_tensor(&bys).sum(Kind::Int64).int64_value(&[]);
        tot_seen += bys.size()[0];
    }
    let acc = if tot_seen > 0 { tot_correct as f64 / tot_seen as f64 } else { 0.0 };
    let denom = (tot_seen as f64 / batch_size as f64).max(1.0);
    let loss = if tot_seen > 0 { tot_loss / denom } else { 0.0 };
    (loss, acc)
}

// =============== MAIN ===============
fn main() -> Result<()> {
    let start_time = Instant::now();
    tch::set_num_threads(num_cpus::get() as i32);
    tch::set_num_interop_threads(1);
    tch::manual_seed(42);
    let device = Device::Cpu;
    println!("Using device: {:?} | threads: {}", device, num_cpus::get());

    // Kelas
    let classes = list_classes(DATA_DIR)?;
    let num_classes = classes.len() as i64;

    // Split
    let (train_images, train_labels) = load_split_dataset_with_classes(DATA_DIR, "train", &classes, true)?;
    let (test_images,  test_labels ) = load_split_dataset_with_classes(DATA_DIR, "test",  &classes, false)?;
    let (mut valid_images, mut valid_labels) = load_split_dataset_with_classes(DATA_DIR, "valid", &classes, false)?;
    if valid_images.is_empty() {
        println!("⚠ 'valid' tidak ditemukan/kosong → pakai TEST sebagai validation.\n");
        valid_images = test_images.iter().map(|t| t.shallow_clone()).collect();
        valid_labels = test_labels.clone();
    }

    println!(" Dataset summary:");
    println!("  Train: {} images", train_images.len());
    println!("  Valid: {} images", valid_images.len());
    println!("  Test:  {} images", test_images.len());
    println!("  Classes ({}): {:?}\n", num_classes, classes);

    fs::create_dir_all(MODEL_DIR)?;
    println!("Model will be saved to: {}\n", MODEL_PATH);

    // ===== TRAINING =====
    {
        let mut vs = nn::VarStore::new(device);
        let root = vs.root();
        let base_vs = root.sub("base");
        let (base, last) = mobilenet_v2_base(&base_vs, WIDTH_MULT);
        let head_vs = root.sub("head");
        let head = mobilenet_v2_head(&head_vs, last, num_classes);
        let net = MobileNetV2Split { base, head };

        // progress bar
        let epoch_pb = ProgressBar::new(EPOCHS as u64);
        epoch_pb.set_style(
            ProgressStyle::with_template(" {spinner:.yellow} [Epoch {pos}/{len}] {wide_msg}")
                .unwrap()
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
        );

        // LOAD PRETRAINED (universal mapper)
        if LOAD_PRETRAINED {
            if std::path::Path::new(PRETRAIN_PATH).exists() {
                match load_pretrained_base_only(device, &mut vs, WIDTH_MULT, PRETRAIN_PATH) {
                    Ok(copied) => epoch_pb.println(format!("✓ Loaded pretrained base (copied {copied}) from {}", PRETRAIN_PATH)),
                    Err(e) => epoch_pb.println(format!("⚠ Pretrained load failed: {e}")),
                }
            } else {
                epoch_pb.println(format!("⚠ Pretrained file not found: {}", PRETRAIN_PATH));
            }
        }

        // FREEZE
        if FREEZE_BASE && LOAD_PRETRAINED {
            let vars = vs.variables();
            let mut frozen = 0usize;
            for (name, var) in vars.iter() {
                if name.starts_with("base.") {
                    var.set_requires_grad(false);
                    frozen += 1;
                }
            }
            epoch_pb.println(format!("✓ Frozen base params: {}", frozen));
        } else if FREEZE_BASE && !LOAD_PRETRAINED {
            epoch_pb.println("⚠ FREEZE_BASE aktif tanpa pretrained → dimatikan.");
        }

        // optimizer
        let mut opt = nn::Adam::default().build(&vs, LEARNING_RATE)?;

        println!("Starting training...\n");
        let mut best_valid_acc = 0.0;

        for epoch in 1..=EPOCHS {
            let mut indices: Vec<usize> = (0..train_images.len()).collect();
            indices.shuffle(&mut rand::thread_rng());

            let total_samples = indices.len() as u64;
            let batch_pb = ProgressBar::new(total_samples);
            batch_pb.set_style(
                ProgressStyle::with_template(
                    "  [ep {prefix}] {elapsed_precise} │{bar:48.magenta/blue}│ {percent:>3}% {pos}/{len} • {per_sec} it/s • eta {eta_precise} • {msg}"
                )
                .unwrap()
                .progress_chars("█▓░"),
            );
            batch_pb.set_prefix(epoch.to_string());

            let mut train_loss_sum = 0.0;
            let mut train_correct = 0i64;
            let mut seen: u64 = 0;

            for chunk in indices.chunks(BATCH_SIZE) {
                let (bxs, bys) = batch_from_vec(&train_images, &train_labels, chunk, device);

                let logits = net.forward_t(&bxs, true);
                let loss = logits.cross_entropy_for_logits(&bys);
                opt.backward_step(&loss);

                train_loss_sum += loss.double_value(&[]);
                let preds = logits.argmax(-1, false);
                let batch_correct = preds.eq_tensor(&bys).sum(Kind::Int64).int64_value(&[]) as u64;

                let bs = bys.size()[0] as u64;
                seen += bs;
                train_correct += batch_correct as i64;

                let running_acc  = (train_correct as f64) / (seen as f64) * 100.0;
                let running_loss = train_loss_sum / (seen as f64 / BATCH_SIZE as f64);
                batch_pb.set_message(format!("loss {running_loss:.4} • acc {running_acc:.2}%"));
                batch_pb.inc(bs);
            }
            batch_pb.finish_and_clear();

            let (valid_loss, valid_acc) = eval_stream(&net, &valid_images, &valid_labels, device, BATCH_SIZE);
            epoch_pb.set_message(format!("val_loss {valid_loss:.4} • val_acc {:.2}%", valid_acc * 100.0));
            epoch_pb.inc(1);

            if valid_acc > best_valid_acc {
                best_valid_acc = valid_acc;
                vs.save(MODEL_PATH)?;
                epoch_pb.println(format!("Saved best model (val acc: {:.2}%)", best_valid_acc * 100.0));
            }
        }
        epoch_pb.finish_with_message("training done");
    }

    // ===== TEST =====
    println!("\n Evaluating on test set...");
    let mut vs = nn::VarStore::new(device);
    let base_vs = vs.root().sub("base");
    let (base, last) = mobilenet_v2_base(&base_vs, WIDTH_MULT);
    let head_vs = vs.root().sub("head");
    let head = mobilenet_v2_head(&head_vs, last, num_classes);
    let net = MobileNetV2Split { base, head };

    vs.load(MODEL_PATH)?;
    let (test_loss, test_acc) = eval_stream(&net, &test_images, &test_labels, device, BATCH_SIZE);
    println!("✓ Test Loss: {:.4} | Test Accuracy: {:.2}%", test_loss, test_acc * 100.0);
    println!("\n⏱ Total time: {:.2}s", start_time.elapsed().as_secs_f64());
    Ok(())
}
