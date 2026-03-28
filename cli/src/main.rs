use clap::{Parser, Subcommand};

mod config;
mod data;
mod run;
mod log;

#[derive(Parser)]
#[command(name = "nl_hecate", version, about = "NL-Hecate: Nested Learning CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Process tokens — training or inference determined by config
    Run {
        /// Path to JSON config file
        #[arg(short, long)]
        config: String,

        /// Override GPU index (CUDA_VISIBLE_DEVICES)
        #[arg(long)]
        gpu: Option<usize>,

        /// Resume from last checkpoint in run_dir
        #[arg(long)]
        resume: bool,
    },

    /// Inspect a checkpoint file
    Inspect {
        /// Path to .safetensors checkpoint
        path: String,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run { config, gpu, resume } => {
            if let Some(gpu_id) = gpu {
                std::env::set_var("CUDA_VISIBLE_DEVICES", gpu_id.to_string());
            }
            run::run(&config, resume);
        }
        Commands::Inspect { path } => {
            inspect(&path);
        }
    }
}

fn inspect(path: &str) {
    use nl_hecate_core::checkpoint::load_stacked_safetensors;
    use std::path::Path;

    let p = Path::new(path);
    if !p.exists() {
        eprintln!("ERROR: file not found: {path}");
        std::process::exit(1);
    }

    match load_stacked_safetensors(p) {
        Ok((params, config, n_blocks, build_state)) => {
            let d = config.swa.d_model;
            let v = config.swa.vocab_size;
            let k = config.k;
            let nh = config.swa.num_heads;
            let total_params: usize = params.num_params();

            println!("============================================================");
            println!("NL-Hecate Checkpoint: {path}");
            println!("============================================================");
            println!("  Model:    d={d}, heads={nh}, vocab={v}");
            println!("  CMS:      k={k}, chunks={:?}", config.chunk_sizes);
            println!("  Blocks:   {n_blocks}");
            println!("  Params:   {total_params:>12}");
            println!("  Rule:     {:?}", config.memory_rule);
            println!("  Comp:     {:?}", config.composition);

            if let Some(bs) = build_state {
                println!("  Step:     {}", bs.global_step);
            } else {
                println!("  Step:     (no build state)");
            }
            println!("============================================================");
        }
        Err(e) => {
            eprintln!("ERROR: failed to load checkpoint: {e}");
            std::process::exit(1);
        }
    }
}
