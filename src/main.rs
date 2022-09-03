use std::io::prelude::*;

use std::fs::File;
use std::io;
use std::io::BufReader;
use std::path::PathBuf;

use clap::Parser;

use buscaluso::*;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    /// Word to search for
    #[clap(value_parser)]
    word: Option<String>,

    /// Rules file
    #[clap(short, long, value_parser)]
    rules: PathBuf,

    /// Dictionary file
    #[clap(short, long, value_parser)]
    dict: PathBuf,

    /// Turn on verbose output
    #[clap(short, long, action = clap::ArgAction::Count)]
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
        for result in cfg.search(&word)?.flatten() {
            let (word, cost) = result;
            println!("{} ({})", word, cost);
        }
    } else {
        // interactive mode
        let mut iter: Option<Busca> = None;
        let mut line = String::new();
        while io::stdin().read_line(&mut line)? > 0 {
            let word = line.trim();
            if cli.verbose >= 2 {
                eprintln!("{:?}", word);
            }
            if !word.is_empty() {
                iter = Some(cfg.search(word)?);
            }
            line.clear();
            if let Some(ref mut iter) = iter {
                if let Some((word, cost)) = iter.flatten().next() {
                    println!("{} ({})", word, cost);
                } else if cli.verbose >= 2 {
                    println!("(done)");
                }
            }
        }
    }
    Ok(())
}
