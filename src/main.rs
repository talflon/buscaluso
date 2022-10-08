// Copyright © 2022 Daniel Getz
// SPDX-License-Identifier: MIT

use std::io::prelude::*;

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::{io, process};

use clap::Parser;

use buscaluso::*;

shadow_rs::shadow!(build);

#[derive(Parser)]
#[clap(author, version, long_version = build::CLAP_LONG_VERSION, about, long_about = None)]
struct Cli {
    /// Word to search for
    word: Option<String>,

    /// Rules file
    #[arg(short, long)]
    rules: PathBuf,

    /// Dictionary file
    #[arg(short, long)]
    dict: PathBuf,

    /// Turn on verbose output
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut cfg = BuscaCfg::new();

    if cli.verbose > 0 {
        eprint!("Loading rules from {:?}...", cli.rules);
        io::stderr().flush()?;
    }
    cfg.load_rules(BufReader::new(File::open(cli.rules)?))?;
    if cli.verbose > 0 {
        eprintln!("done");
        if cli.verbose >= 2 {
            eprintln!("{:?}", cfg);
        }
        eprint!("Loading dictionary from {:?}...", cli.dict);
        io::stderr().flush()?;
    }
    cfg.load_dictionary(BufReader::new(File::open(cli.dict)?))?;
    if cli.verbose > 0 {
        eprintln!("done");
    }

    if let Some(word) = cli.word {
        let mut busca = cfg.search(&word)?;
        let mut debugger = BuscaDebugger::new();
        while let Some(result) = busca.iter().next() {
            if let Some((word, cost)) = result {
                println!("{} ({})", word, cost);
            }
            if cli.verbose >= 3 {
                debugger.print_new_int_reps(&busca);
            }
        }
    } else {
        // interactive mode

        let running = Arc::new(AtomicBool::new(false));
        {
            let running = Arc::clone(&running);
            ctrlc::set_handler(move || {
                println!();
                if !running.swap(false, Ordering::SeqCst) {
                    // force quit on second Ctrl-C without being cleared
                    process::exit(1)
                }
            })
            .unwrap();
        }

        let mut busca: Option<Busca> = None;
        let mut line = String::new();
        let mut debugger: Option<BuscaDebugger> = None;
        while io::stdin().read_line(&mut line)? > 0 {
            let word = line.trim();
            if cli.verbose >= 2 {
                eprintln!("{:?}", word);
            }
            if !word.is_empty() {
                busca = Some(cfg.search(word)?);
                if cli.verbose >= 3 {
                    debugger = Some(BuscaDebugger::new());
                };
            }
            line.clear();

            if let Some(ref mut busca) = busca {
                running.store(true, Ordering::SeqCst);
                loop {
                    match busca.iter().next() {
                        Some(Some((word, cost))) => {
                            println!("{} ({})", word, cost);
                            break;
                        }
                        Some(None) => {
                            if !running.load(Ordering::SeqCst) {
                                break;
                            }
                        }
                        None => {
                            if cli.verbose >= 2 {
                                println!("(done)");
                            }
                            break;
                        }
                    }
                }
                running.store(false, Ordering::SeqCst);

                if cli.verbose >= 3 {
                    if let Some(ref mut debugger) = debugger {
                        debugger.print_new_int_reps(busca);
                    }
                }
            }
        }
    }
    Ok(())
}
