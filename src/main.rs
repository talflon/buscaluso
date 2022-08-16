use std::io;
use std::io::prelude::*;
use std::io::BufReader;
use std::{fs::File, path::PathBuf};

use clap::Parser;

use buscaluso::*;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    /// Word to search for
    #[clap(value_parser)]
    word: String,

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

    for result in cfg.search(&cli.word) {
        let (word, cost) = result?;
        println!("{} ({})", word, cost);
    }
    Ok(())
}
