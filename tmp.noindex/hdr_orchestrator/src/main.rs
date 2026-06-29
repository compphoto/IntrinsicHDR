use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

// Default paths
const DEFAULT_SDR_DIR: &str = "/Users/iphoobis/Desktop/SDR";
const DEFAULT_HDR_DIR: &str = "/Users/iphoobis/Desktop/HDR";
const PROJECT_DIR: &str = "/Users/iphoobis/Projects/IntrinsicHDR";
const PYTHON_BIN: &str = "/opt/homebrew/Caskroom/miniforge/base/envs/intrinsic-hdr/bin/python";

fn print_help() {
    println!("HDR Orchestrator Binary (Batch Optimized)");
    println!("Usage: hdr_orchestrator [OPTIONS]");
    println!("\nOptions:");
    println!("  --use_exr        Export processed images as '.exr' instead of '.hdr'");
    println!("  --sdr <path>     Custom path to SDR directory (default: ~/Desktop/SDR)");
    println!("  --hdr <path>     Custom path to HDR directory (default: ~/Desktop/HDR)");
    println!("  --res <val>      Decomposition resolution limit (default: 1536, set to 'none' to disable)");
    println!("  --batch_size <n> Number of images to process in a single PyTorch session (default: 100)");
    println!("  -h, --help       Print this help message");
}

fn main() -> io::Result<()> {
    // 1. Parse arguments
    let args: Vec<String> = env::args().collect();
    let mut use_exr = false;
    let mut sdr_path_str = DEFAULT_SDR_DIR.to_string();
    let mut hdr_path_str = DEFAULT_HDR_DIR.to_string();
    let mut res_limit = Some(1536); // Tuned for 32GB unified memory (M1 Pro); ~1.4x long side over the 16GB-safe 1024
    let mut batch_size: usize = 100;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--use_exr" => {
                use_exr = true;
            }
            "--sdr" => {
                if i + 1 < args.len() {
                    sdr_path_str = args[i + 1].clone();
                    i += 1;
                } else {
                    eprintln!("Error: --sdr requires a path argument.");
                    print_help();
                    std::process::exit(1);
                }
            }
            "--hdr" => {
                if i + 1 < args.len() {
                    hdr_path_str = args[i + 1].clone();
                    i += 1;
                } else {
                    eprintln!("Error: --hdr requires a path argument.");
                    print_help();
                    std::process::exit(1);
                }
            }
            "--res" => {
                if i + 1 < args.len() {
                    let val = args[i + 1].clone();
                    if val.to_lowercase() == "none" {
                        res_limit = None;
                    } else if let Ok(num) = val.parse::<u32>() {
                        res_limit = Some(num);
                    } else {
                        eprintln!("Error: --res must be an integer (e.g. 1024, 2048) or 'none'.");
                        std::process::exit(1);
                    }
                    i += 1;
                } else {
                    eprintln!("Error: --res requires a value.");
                    print_help();
                    std::process::exit(1);
                }
            }
            "--batch_size" => {
                if i + 1 < args.len() {
                    if let Ok(num) = args[i + 1].parse::<usize>() {
                        batch_size = num;
                    } else {
                        eprintln!("Error: --batch_size must be a positive integer.");
                        std::process::exit(1);
                    }
                    i += 1;
                } else {
                    eprintln!("Error: --batch_size requires a value.");
                    print_help();
                    std::process::exit(1);
                }
            }
            "-h" | "--help" => {
                print_help();
                return Ok(());
            }
            arg => {
                eprintln!("Error: Unknown argument '{}'", arg);
                print_help();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let sdr_dir = PathBuf::from(sdr_path_str);
    let hdr_dir = PathBuf::from(hdr_path_str);
    let db_path = hdr_dir.join(".processed_db.txt");

    // 2. Initialize environment
    println!("Initializing environment...");

    // Clear any stale pause file left over from a previously killed run,
    // otherwise the conversion/inference phases would block immediately.
    let pause_file = PathBuf::from(env::var("HOME").unwrap_or_else(|_| "/Users/iphoobis".to_string()))
        .join(".hdr_pause");
    if pause_file.exists() {
        let _ = fs::remove_file(&pause_file);
        println!("Cleared stale pause file from a previous run.");
    }
    if !sdr_dir.exists() {
        println!("Creating SDR directory at {:?}", sdr_dir);
        fs::create_dir_all(&sdr_dir)?;
    }
    if !hdr_dir.exists() {
        println!("Creating HDR directory at {:?}", hdr_dir);
        fs::create_dir_all(&hdr_dir)?;
    }

    // Load database
    let mut processed_set = HashSet::new();
    if db_path.exists() {
        let file = fs::File::open(&db_path)?;
        let reader = BufReader::new(file);
        for line in reader.lines() {
            if let Ok(filename) = line {
                let trimmed = filename.trim();
                if !trimmed.is_empty() {
                    processed_set.insert(trimmed.to_string());
                }
            }
        }
    }
    println!("Loaded {} already-processed files from database.", processed_set.len());

    // 3. Set Ctrl+C Handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        println!("\n[Ctrl+C] Interruption detected! Harvesting completed files from current batch and exiting cleanly...");
        r.store(false, Ordering::SeqCst);
    }).expect("Error setting Ctrl+C handler");

    // Hotkey Pause Listener (p + Enter)
    std::thread::spawn(|| {
        let stdin = io::stdin();
        let mut handle = stdin.lock();
        let mut buffer = String::new();
        while handle.read_line(&mut buffer).is_ok() {
            let input = buffer.trim();
            if input.eq_ignore_ascii_case("p") {
                let pause_file = PathBuf::from(env::var("HOME").unwrap_or("/Users/iphoobis".to_string())).join(".hdr_pause");
                if pause_file.exists() {
                    let _ = fs::remove_file(&pause_file);
                    println!("\n[Hotkey] Unpause triggered via 'p'. Resuming...");
                } else {
                    let _ = fs::File::create(&pause_file);
                    println!("\n[Hotkey] Pause triggered via 'p'. Will pause and release memory after current image finishes.");
                }
            }
            buffer.clear();
        }
    });

    // 4. Scan SDR directory
    let mut image_files = Vec::new();
    for entry in fs::read_dir(&sdr_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                let ext_lower = ext.to_lowercase();
                if matches!(
                    ext_lower.as_str(),
                    "png" | "jpg" | "jpeg" | "tiff" | "tif" | "webp" | "jxl" | "avif"
                ) {
                    if let Some(filename) = path.file_name().and_then(|f| f.to_str()) {
                        image_files.push((filename.to_string(), path.clone()));
                    }
                }
            }
        }
    }

    // Sort files to process in a predictable sequential order
    image_files.sort_by(|a, b| a.0.cmp(&b.0));

    // Filter out already processed files
    let pending_files: Vec<(String, PathBuf)> = image_files
        .into_iter()
        .filter(|(filename, _)| !processed_set.contains(filename))
        .collect();

    let total_pending = pending_files.len();
    if total_pending == 0 {
        println!("No new images to process in SDR folder: {:?}", sdr_dir);
        return Ok(());
    }

    println!("Found {} new images to process in SDR directory.", total_pending);
    println!("Batch size is set to {} images per PyTorch session.", batch_size);
    if let Some(res) = res_limit {
        println!("Decomposition resolution limit set to {}px.", res);
    } else {
        println!("Decomposition resolution limit is disabled.");
    }

    // Temp root directory
    let temp_root = Path::new(PROJECT_DIR).join("tmp.noindex").join("rust_orchestrator");
    fs::create_dir_all(&temp_root)?;

    // 5. Processing in batches
    let mut processed_count = 0;
    let chunks = pending_files.chunks(batch_size);
    let total_batches = (total_pending as f32 / batch_size as f32).ceil() as usize;

    for (batch_idx, batch) in chunks.enumerate() {
        if !running.load(Ordering::SeqCst) {
            println!("Workflow cancelled by user.");
            break;
        }

        println!("\n==================================================");
        println!("PROCESSING BATCH {}/{} ({} images)", batch_idx + 1, total_batches, batch.len());
        println!("==================================================");

        // Subfolder setup for this batch
        let batch_run_id = format!("batch_{}", batch_idx);
        let run_temp_dir = temp_root.join(&batch_run_id);
        let temp_in_dir = run_temp_dir.join("inputs");
        let temp_out_dir = run_temp_dir.join("outputs");

        fs::create_dir_all(&temp_in_dir)?;
        fs::create_dir_all(&temp_out_dir)?;

        // Map to keep track of stems -> original names and paths
        let mut stem_map = HashMap::new();
        let mut conversion_tasks = Vec::new();

        // Convert all SDR images in the batch to linear EXR
        println!("Step 1/2: Converting {} images to linear EXR files in parallel...", batch.len());
        let mut conversion_failed = false;
        
        for (filename, path) in batch {
            let file_stem = path.file_stem().unwrap().to_str().unwrap().to_string();
            
            // Handle duplicate stems in the same batch by appending an index
            let mut unique_stem = file_stem.clone();
            let mut counter = 1;
            while stem_map.contains_key(&unique_stem) {
                unique_stem = format!("{}_{}", file_stem, counter);
                counter += 1;
            }

            stem_map.insert(unique_stem.clone(), (filename.clone(), path.clone()));

            let temp_exr_path = temp_in_dir.join(format!("{}.exr", unique_stem));
            conversion_tasks.push((filename.clone(), path.clone(), temp_exr_path));
        }

        if running.load(Ordering::SeqCst) {
            let num_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
            for chunk in conversion_tasks.chunks(num_threads) {
                if !running.load(Ordering::SeqCst) {
                    break;
                }

                // Pause check
                let pause_file = PathBuf::from(env::var("HOME").unwrap_or("/Users/iphoobis".to_string())).join(".hdr_pause");
                if pause_file.exists() {
                    println!("\n[Pause] Conversion phase paused. Waiting for {} to be removed...", pause_file.display());
                    while pause_file.exists() {
                        std::thread::sleep(std::time::Duration::from_secs(2));
                        if !running.load(Ordering::SeqCst) {
                            break;
                        }
                    }
                    if !running.load(Ordering::SeqCst) {
                        break;
                    }
                    println!("[Pause] Resuming conversion...");
                }

                let mut children = Vec::new();
                for (filename, path, temp_exr_path) in chunk {
                    let child = Command::new(PYTHON_BIN)
                        .arg(format!("{}/prepare_input.py", PROJECT_DIR))
                        .arg(path)
                        .arg(temp_exr_path)
                        .env("OPENCV_IO_ENABLE_OPENEXR", "1")
                        .stdin(Stdio::null())
                        .spawn();
                        
                    match child {
                        Ok(c) => children.push((filename.clone(), c)),
                        Err(e) => {
                            eprintln!("[Error] Failed to spawn conversion for {}: {}", filename, e);
                            conversion_failed = true;
                        }
                    }
                }

                for (filename, mut child) in children {
                    match child.wait() {
                        Ok(status) if status.success() => {}
                        _ => {
                            eprintln!("[Error] Conversion failed for image {}", filename);
                            conversion_failed = true;
                        }
                    }
                }

                if conversion_failed {
                    break;
                }
            }
        }

        if conversion_failed || !running.load(Ordering::SeqCst) {
            let _ = fs::remove_dir_all(&run_temp_dir);
            if !running.load(Ordering::SeqCst) {
                println!("Batch execution aborted by user command.");
                break;
            }
            continue;
        }

        // Step 2: Run batch inference
        println!("Step 2/2: Starting PyTorch MPS session for batch of {} images...", batch.len());
        let mut inf_cmd = Command::new(PYTHON_BIN);
        inf_cmd.arg(format!("{}/inference.py", PROJECT_DIR))
            .arg("--test_imgs")
            .arg(&temp_in_dir)
            .arg("--output_path")
            .arg(&temp_out_dir)
            .env("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            .env("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.84")
            .env("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.7")
            .env("OPENCV_IO_ENABLE_OPENEXR", "1");

        if use_exr {
            inf_cmd.arg("--use_exr");
        }

        if let Some(res) = res_limit {
            inf_cmd.arg("--res").arg(res.to_string());
        }

        // Keep terminal stdin reserved for the parent's hotkey listener.
        inf_cmd.stdin(Stdio::null());

        // Wait for inference to complete (or be interrupted)
        let _ = inf_cmd.status();

        // Harvesting phase (Runs even if inference.py failed/interrupted, to save partially processed files!)
        println!("Harvesting processed images from current batch...");
        let pred_dir = temp_out_dir.join("predictions");
        let mut batch_harvest_count = 0;

        if pred_dir.exists() {
            if let Ok(entries) = fs::read_dir(pred_dir) {
                for entry in entries.flatten() {
                    let pred_path = entry.path();
                    if pred_path.is_file() {
                        let stem = pred_path.file_stem().unwrap().to_str().unwrap();
                        if let Some((orig_filename, orig_path)) = stem_map.get(stem) {
                            let out_ext = if use_exr { "exr" } else { "hdr" };
                            let final_hdr_path = hdr_dir.join(format!("{}.{}", stem, out_ext));

                            // Move to final HDR directory
                            if fs::rename(&pred_path, &final_hdr_path).is_ok() {
                                // Log in database
                                if let Ok(mut db_file) = OpenOptions::new().create(true).append(true).open(&db_path) {
                                    if writeln!(db_file, "{}", orig_filename).is_ok() {
                                        processed_set.insert(orig_filename.clone());
                                    }
                                }

                                // Delete original SDR file
                                let _ = fs::remove_file(orig_path);
                                batch_harvest_count += 1;
                            }
                        }
                    }
                }
            }
        }

        println!("Successfully finalized and saved {} images from this batch.", batch_harvest_count);
        processed_count += batch_harvest_count;

        // Clean up temp directories
        let _ = fs::remove_dir_all(&run_temp_dir);
    }

    // Try to remove temp root if empty
    let _ = fs::remove_dir(&temp_root);

    println!("\nWorkflow completed. Processed {} new images.", processed_count);
    Ok(())
}
