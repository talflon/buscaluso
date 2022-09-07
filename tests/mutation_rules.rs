use buscaluso::*;
use ntest::timeout;

struct SearchTracker<'a> {
    past: Vec<&'a str>,
    future: Busca<'a>,
}

impl<'a> SearchTracker<'a> {
    fn finds(&mut self, word: &str) -> bool {
        if self.past.contains(&word) {
            return true;
        }
        while let Some((s, _c)) = self.future.by_ref().flatten().next() {
            self.past.push(s);
            if s == word {
                return true;
            }
        }
        false
    }

    fn all(&mut self) -> &[&'a str] {
        while let Some((s, _c)) = self.future.by_ref().flatten().next() {
            self.past.push(s);
        }
        &self.past
    }

    fn assert_finds(&mut self, word: &str) {
        assert!(
            self.finds(word),
            "Didn't find {:?} (found only: {:?})",
            word,
            self.all()
        );
    }

    fn assert_finds_all<'b>(&mut self, words: impl IntoIterator<Item = &'b str>) {
        for word in words {
            self.assert_finds(word);
        }
    }

    fn assert_finds_not(&mut self, word: &str) {
        assert!(
            !self.all().contains(&word),
            "Found {:?} when we were not supposed to",
            word
        );
    }
}

impl<'a> From<Busca<'a>> for SearchTracker<'a> {
    fn from(busca: Busca<'a>) -> Self {
        SearchTracker {
            past: Vec::new(),
            future: busca,
        }
    }
}

#[test]
#[timeout(1_000)]
fn simplest_mutation_rule() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.load_rules("1: a > b".as_bytes())?;
    cfg.load_dictionary("b".as_bytes())?;
    let mut tracker = SearchTracker::from(cfg.search("a")?);
    tracker.assert_finds("b");
    Ok(())
}

#[test]
#[timeout(1_000)]
fn mutation_rule_reverse() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.load_rules("1: b > a".as_bytes())?;
    cfg.load_dictionary("b".as_bytes())?;
    let mut tracker = SearchTracker::from(cfg.search("a")?);
    tracker.assert_finds("b");
    Ok(())
}

#[test]
#[timeout(1_000)]
fn test_mutation_rule_applied_multiple_times() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.load_rules("1: a > b".as_bytes())?;
    let dict = vec!["bbbb", "babb", "bbab"];
    cfg.load_dictionary(dict.join("\n").as_bytes())?;
    let mut tracker = SearchTracker::from(cfg.search("baba")?);
    tracker.assert_finds_all(dict);
    Ok(())
}

#[test]
#[timeout(1_000)]
fn test_doesnt_just_return_all_dict_words() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.load_rules("1: a b c > d e f".as_bytes())?;
    let dict = vec!["bbbb", "babb", "bbab"];
    cfg.load_dictionary(dict.join("\n").as_bytes())?;
    assert_eq!(
        SearchTracker::from(cfg.search("baba")?).all(),
        &[] as &[&str]
    );
    Ok(())
}

#[test]
#[timeout(1_000)]
fn test_lookaround_rule() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.load_rules("1: b | i > e | g".as_bytes())?;
    let dict = vec!["big", "beg", "bib"];
    cfg.load_dictionary(dict.join("\n").as_bytes())?;
    let mut tracker = SearchTracker::from(cfg.search("big")?);
    tracker.assert_finds("beg");
    Ok(())
}

#[test]
#[timeout(1_000)]
fn test_two_rules() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.load_rules("1: a > b\n2: c > a".as_bytes())?;
    cfg.load_dictionary("b".as_bytes())?;
    let mut tracker = SearchTracker::from(cfg.search("c")?);
    tracker.assert_finds("b");
    Ok(())
}

#[test]
#[timeout(1_000)]
fn test_empty_replacement_rule() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.load_rules(
        "V = [a e i o u]\n\
        C = [c b n]\n\
    1: V | r > _"
            .as_bytes(),
    )?;
    let dict = vec!["caborn", "car"];
    cfg.load_dictionary(dict.join("\n").as_bytes())?;
    let mut tracker = SearchTracker::from(cfg.search("carbon")?);
    tracker.assert_finds("caborn");
    Ok(())
}

#[test]
#[timeout(1_000)]
fn test_start_anchor_rule() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.load_rules("1: _ | x > y".as_bytes())?;
    let dict = vec!["xx", "xy", "yx", "yy"];
    cfg.load_dictionary(dict.join("\n").as_bytes())?;
    let mut tracker = SearchTracker::from(cfg.search("xx")?);
    tracker.assert_finds("yx");
    Ok(())
}

#[test]
#[timeout(1_000)]
fn test_end_anchor_rule() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.load_rules("1: x > y | _".as_bytes())?;
    let dict = vec!["xx", "xy", "yx", "yy"];
    cfg.load_dictionary(dict.join("\n").as_bytes())?;
    let mut tracker = SearchTracker::from(cfg.search("xx")?);
    tracker.assert_finds("xy");
    Ok(())
}

#[test]
#[timeout(1_000)]
fn test_both_anchor_rule() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.load_rules("1: _ | x > y | _".as_bytes())?;
    let dict = vec!["x", "y"];
    cfg.load_dictionary(dict.join("\n").as_bytes())?;
    let mut tracker = SearchTracker::from(cfg.search("x")?);
    tracker.assert_finds("y");
    Ok(())
}
