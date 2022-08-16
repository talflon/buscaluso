use super::*;

use std::collections::BTreeSet;

use rulefile::Item;

#[test]
fn test_fon_registry_empty() {
    let reg = FonRegistry::new();
    for c in ['q', 'é', '\0'] {
        assert!(matches!(reg.get_id(c), Err(NoSuchFon(c_)) if c_ == c));
    }
    for i in [0, 1, 2, 35, MAX_FON_ID] {
        assert!(matches!(reg.get_fon(i), Err(NoSuchFonId(i_)) if i_ == i));
    }
}

#[test]
fn test_fon_registry_add() -> Result<()> {
    let mut reg = FonRegistry::new();
    for c in ['Z', 'Ç'] {
        reg.add(c)?;
        assert_eq!(reg.get_fon(reg.get_id(c)?)?, c);
    }
    Ok(())
}

#[test]
fn test_fon_registry_add_same() -> Result<()> {
    let mut reg = FonRegistry::new();
    let c = '$';
    let id = reg.add(c)?;
    for _ in 0..10000 {
        assert_eq!(reg.add(c)?, id);
    }
    Ok(())
}

#[test]
fn test_too_many_registries() -> Result<()> {
    let mut reg = FonRegistry::new();
    for c in (1..)
        .flat_map(char::from_u32)
        .take((MAX_FON_ID as usize) + 1)
    {
        reg.add(c)?;
    }
    assert!(matches!(reg.add('\0'), Err(NoMoreFonIds)));
    Ok(())
}

#[test]
fn test_fonset_null() {
    let s = FonSet::NULL;
    assert!(s.is_empty());
    assert_eq!(s.len(), 0);
    for i in 0..=MAX_FON_ID {
        assert!(!s.contains(i))
    }
}

#[test]
fn test_fonset_default() {
    let s: FonSet = Default::default();
    assert_eq!(s, FonSet::NULL);
}

#[test]
fn test_fonset_from_one_contains_it() {
    for i in 0..=MAX_FON_ID {
        assert!(FonSet::from(i).contains(i));
    }
}

#[test]
fn test_fonset_from_one_only_contains_it() {
    for i in 0..=MAX_FON_ID {
        let s = FonSet::from(i);
        assert_eq!(s.len(), 1);
        assert!(!s.is_empty());
        for j in 0..=MAX_FON_ID {
            if i != j {
                assert!(!s.contains(j));
            }
        }
    }
}

#[test]
fn test_fonset_add_with_oreq() {
    let mut s = FonSet::NULL;
    s |= 3;
    s |= 17;
    s |= 1;
    s |= 0;
    s |= 4;
    assert_eq!(s.len(), 5);
}

#[test]
fn test_fonset_subtract() {
    let s = FonSet::from([100, 99]);
    assert_eq!(s - 99, FonSet::from(100));
}

#[test]
fn test_oreq_subeq() {
    let mut s = FonSet::NULL;
    s |= 77;
    s -= 77;
    assert!(!s.contains(77));
}

#[test]
fn test_fonset_from_to_iter() {
    use std::collections::HashSet;
    let hs: HashSet<FonId> = [50, 7, 12, 4].into();
    let fs: FonSet = hs.iter().cloned().collect();
    let fshs: HashSet<FonId> = fs.iter().collect();
    assert_eq!(fshs, hs);
    assert!(fs.contains(12));
}

#[test]
fn test_fonset_iter() {
    for i in 0..=MAX_FON_ID {
        let v: Vec<FonId> = FonSet::from(i).iter().collect();
        assert_eq!(v, vec![i]);
    }
}

#[test]
fn test_fonset_for_loop() {
    let s1 = FonSet::from([0, 55, 3, 11, 8]);
    let mut s2: FonSet = FonSet::NULL;
    for i in s1 {
        s2 |= i;
    }
    assert_eq!(s2, s1);
}

#[test]
fn test_fonset_fons() -> Result<()> {
    let mut reg = FonRegistry::new();
    let mut s = FonSet::NULL;
    let chars = vec!['$', 'q', 'A', 'ç'];
    for &c in chars.iter() {
        s |= reg.add(c)?;
    }
    assert_eq!(s.fons(&reg)?, chars);
    assert_eq!(FonSet::NULL.fons(&reg)?, vec![]);
    Ok(())
}

#[test]
fn test_fonsetseq_empty() {
    assert!(FonSetSeq::from([]).is_empty());
    assert!(FonSetSeq::from([FonSet::NULL]).is_empty());
    assert!(FonSetSeq::from([FonSet::NULL, FonSet::NULL]).is_empty());
    assert!(FonSetSeq::from([FonSet::NULL, FonSet::from(3)]).is_empty());
    assert!(FonSetSeq::from([2.into(), FonSet::NULL]).is_empty());
    assert!(!FonSetSeq::from([2.into(), 3.into()]).is_empty());
}

#[test]
fn test_fonsetseq_match_at_in_bounds() {
    let seq1 = FonSetSeq::from([[2, 4].into(), [1, 2, 3].into()]);
    let seq2 = FonSetSeq::from([[2, 5].into()]);
    assert_eq!(
        seq2.match_at(&seq1, 0),
        Some(FonSetSeq::from([2.into(), [1, 2, 3].into()]))
    );
    assert_eq!(
        seq2.match_at(&seq1, 1),
        Some(FonSetSeq::from([[2, 4].into(), 2.into()]))
    );
    assert_eq!(FonSetSeq::from([81.into()]).match_at(&seq1, 0), None);
    assert_eq!(FonSetSeq::from([81.into()]).match_at(&seq1, 1), None);
}

#[test]
fn test_fonsetseq_match_at_out_of_bounds() {
    assert_eq!(FonSetSeq::from([]).match_at(&FonSetSeq::from([]), 1), None);
    assert_eq!(
        FonSetSeq::from([[32, 0].into()]).match_at(&FonSetSeq::from([[90, 0, 1].into()]), 1),
        None
    );
}

#[test]
fn test_registry_normalize() -> Result<()> {
    let mut reg = FonRegistry::new();
    let s1 = "test";
    let s2 = "blah";
    for c in s1.chars().chain(s2.chars()) {
        reg.add(c)?;
    }
    let mut n1a = Vec::new();
    reg.normalize_into(s1, &mut n1a)?;
    assert!(!n1a.is_empty());
    let mut n1b = Vec::new();
    reg.normalize_into(&"a test!"[2..6], &mut n1b)?;
    assert_eq!(n1b, n1a);
    let mut n2 = Vec::new();
    reg.normalize_into(s2, &mut n2)?;
    assert_ne!(n2, n1a);
    Ok(())
}

#[test]
fn test_registry_normalize_bad_chars() {
    let reg = FonRegistry::new();
    assert!(matches!(reg.normalize("hi"), Err(NoSuchFon(_))));
}

#[test]
fn test_registry_normalize_lowercases() -> Result<()> {
    let mut reg = FonRegistry::new();
    for c in "great".chars() {
        reg.add(c)?;
    }
    assert_eq!(reg.normalize("GrEat")?, reg.normalize("great")?);
    Ok(())
}

#[test]
fn test_dictionary_one_key() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    assert_eq!(Vec::from_iter(cfg.words_iter(b"one")), &[] as &[&str]);

    cfg.add_to_dictionary("first", b"one")?;
    assert_eq!(Vec::from_iter(cfg.words_iter(b"one")), &["first"]);

    cfg.add_to_dictionary("another", b"one")?;
    assert_eq!(
        BTreeSet::from_iter(cfg.words_iter(b"one")),
        BTreeSet::from(["another", "first"])
    );
    Ok(())
}

#[test]
fn test_dictionary_two_keys() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.add_to_dictionary("first", b"one")?;
    cfg.add_to_dictionary("another", b"one")?;
    assert_eq!(Vec::from_iter(cfg.words_iter(b"two")), &[] as &[&str]);

    cfg.add_to_dictionary("word", b"two")?;
    assert_eq!(Vec::from_iter(cfg.words_iter(b"two")), &["word"]);

    assert_eq!(
        BTreeSet::from_iter(cfg.words_iter(b"one")),
        BTreeSet::from(["another", "first"])
    );
    Ok(())
}

#[test]
fn test_dictionary_duplicate_one_key() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    assert_eq!(Vec::from_iter(cfg.words_iter(b"key")), &[] as &[&str]);

    cfg.add_to_dictionary("value", b"key")?;
    assert_eq!(Vec::from_iter(cfg.words_iter(b"key")), &["value"]);
    cfg.add_to_dictionary("value", b"key")?;
    assert_eq!(Vec::from_iter(cfg.words_iter(b"key")), &["value"]);
    Ok(())
}

#[test]
fn test_dictionary_duplicate_two_keys() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.add_to_dictionary("same", b"one")?;
    cfg.add_to_dictionary("same", b"two")?;
    assert_eq!(Vec::from_iter(cfg.words_iter(b"one")), &["same"]);
    assert_eq!(Vec::from_iter(cfg.words_iter(b"two")), &["same"]);
    Ok(())
}

#[test]
fn test_empty_ruleset() {
    let ruleset = NormalizeRuleSet::new();
    assert_eq!(ruleset.longest_rule(), 0);
    assert_eq!(ruleset.get_rule(&['a']), None);
    assert_eq!(ruleset.get_rule(&['b', 'c']), None);
    assert_eq!(ruleset.find_rule(&['d']), None);
    assert_eq!(ruleset.find_rule(&['e', 'f']), None);
}

#[test]
fn test_ruleset_add_get() -> Result<()> {
    let mut ruleset = NormalizeRuleSet::new();
    ruleset.add_rule(&['c'], b"yes")?;
    assert_eq!(ruleset.longest_rule(), 1);
    ruleset.add_rule(&['a', 'b'], b"ok")?;
    assert_eq!(ruleset.longest_rule(), 2);
    assert_eq!(ruleset.get_rule(&['a']), None);
    assert_eq!(ruleset.get_rule(&['a', 'b']), Some(&b"ok"[..]));
    assert_eq!(ruleset.get_rule(&['c']), Some(&b"yes"[..]));
    assert_eq!(ruleset.get_rule(&['x', 'c']), None);
    assert_eq!(ruleset.get_rule(&['c', 'x']), None);
    assert_eq!(ruleset.get_rule(&['d']), None);
    Ok(())
}

#[test]
fn test_ruleset_add_duplicate() -> Result<()> {
    let mut ruleset = NormalizeRuleSet::new();
    ruleset.add_rule(&['c'], b"yes")?;
    assert!(matches!(
        ruleset.add_rule(&['c'], b"no"),
        Err(DuplicateNormRule(_))
    ));
    ruleset.add_rule(&['a', 'b'], b"ok")?;
    assert!(matches!(
        ruleset.add_rule(&['c'], b"no"),
        Err(DuplicateNormRule(_))
    ));
    assert!(matches!(
        ruleset.add_rule(&['a', 'b'], b"nope"),
        Err(DuplicateNormRule(_))
    ));
    Ok(())
}

#[test]
fn test_ruleset_find() -> Result<()> {
    let mut ruleset = NormalizeRuleSet::new();
    ruleset.add_rule(&['a', 'b'], b"ok")?;
    ruleset.add_rule(&['c'], b"yes")?;
    assert_eq!(ruleset.find_rule(&['x']), None);
    assert_eq!(ruleset.find_rule(&['x', 'a', 'b', 'y']), None);
    assert_eq!(ruleset.find_rule(&['x', 'c', 'y']), None);
    assert_eq!(ruleset.find_rule(&['c']), Some((1, &b"yes"[..])));
    assert_eq!(ruleset.find_rule(&['c', 'z']), Some((1, &b"yes"[..])));
    assert_eq!(ruleset.find_rule(&['a', 'b']), Some((2, &b"ok"[..])));
    assert_eq!(ruleset.find_rule(&['a', 'b', 'n']), Some((2, &b"ok"[..])));
    Ok(())
}

#[test]
fn test_normalize_ruleset_and_reg() -> Result<()> {
    let mut ruleset = NormalizeRuleSet::new();
    let mut reg = FonRegistry::new();
    for c in "mno".chars() {
        reg.add(c)?;
    }
    ruleset.add_rule(&['a', 'b', 'c'], &[reg.add('z')?])?;
    assert_eq!(
        (&ruleset, &reg).normalize("noabcm")?,
        reg.normalize("nozm")?
    );
    Ok(())
}

#[test]
fn test_normalize_ruleset_and_reg_empty_rule() -> Result<()> {
    let mut ruleset = NormalizeRuleSet::new();
    let mut reg = FonRegistry::new();
    for c in "normal".chars() {
        reg.add(c)?;
    }
    ruleset.add_rule(&['a', 'b', 'c'], &[])?;
    ruleset.add_rule(&['e'], &[])?;
    assert_eq!(
        (&ruleset, &reg).normalize("noabcrmael")?,
        reg.normalize("normal")?
    );
    Ok(())
}

#[test]
fn test_normalize_ruleset_and_reg_lowercases() -> Result<()> {
    let ruleset = NormalizeRuleSet::new();
    let mut reg = FonRegistry::new();
    for c in "great".chars() {
        reg.add(c)?;
    }
    assert_eq!(
        (&ruleset, &reg).normalize("GrEat")?,
        (&ruleset, &reg).normalize("great")?
    );
    Ok(())
}

#[test]
fn test_buscacfg_normalize() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    for c in "mno".chars() {
        cfg.fon_registry.add(c)?;
    }
    cfg.normalize_rules
        .add_rule(&['a', 'b', 'c'], &[cfg.fon_registry.add('z')?])?;
    assert_eq!(
        cfg.normalize("noabcm")?,
        cfg.fon_registry.normalize("nozm")?
    );
    Ok(())
}

#[test]
fn test_buscacfg_search() -> Result<()> {
    let s = "word";
    let mut cfg = BuscaCfg::new();
    for c in s.chars() {
        cfg.fon_registry.add(c)?;
    }
    cfg.add_to_dictionary(s, &cfg.normalize(s)?)?;
    let result: Result<Vec<(&str, Cost)>> = cfg.search(s).collect();
    assert_eq!(result?, vec![(s, 0)]);
    Ok(())
}

#[test]
fn test_buscacfg_search_normalize_error() {
    let cfg = BuscaCfg::new();
    assert!(matches!(cfg.search("anything").next(), Some(Err(_))));
}

#[test]
fn test_set_alias() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    let fons = [3, 4, 5];
    assert_eq!(cfg.try_get_alias("blah"), None);
    assert!(matches!(cfg.get_alias("blah"), Err(NoSuchAlias(_))));
    cfg.set_alias("blah".into(), fons.into());
    assert_eq!(cfg.try_get_alias("blah"), Some(fons.into()));
    assert_eq!(cfg.get_alias("blah")?, fons.into());
    assert_eq!(cfg.try_get_alias("blah!"), None);
    assert!(matches!(cfg.get_alias("blah!"), Err(NoSuchAlias(_))));
    Ok(())
}

#[test]
fn test_add_rule_alias() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.fon_registry.add('a')?;
    cfg.add_rule(Rule::Alias("Hi", vec![Item::Char('a')]))?;
    assert_eq!(
        cfg.get_alias("Hi")?,
        FonSet::from(cfg.fon_registry.get_id('a')?)
    );
    Ok(())
}
