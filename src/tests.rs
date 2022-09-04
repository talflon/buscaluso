use super::*;

use std::collections::BTreeSet;

use quickcheck::{Arbitrary, TestResult};
use quickcheck_macros::*;

use rulefile::Item;

#[test]
fn test_fon_registry_empty() {
    let reg = FonRegistry::new();
    for c in ['q', 'é', '\0'] {
        assert!(matches!(reg.get_id(c), Err(NoSuchFon(c_)) if c_ == c));
    }
    for i in [1, 2, 35, MAX_FON_ID] {
        assert!(matches!(reg.get_fon(i), Err(NoSuchFonId(i_)) if i_ == i));
    }
}

#[test]
fn test_fon_registry_has_no_fon() -> Result<()> {
    let mut reg = FonRegistry::new();
    assert_eq!(reg.try_get_id(NO_FON_CHAR), Some(NO_FON));
    assert_eq!(reg.try_get_fon(NO_FON), Some(NO_FON_CHAR));
    reg.add('ȟ')?;
    reg.add('r')?;
    assert_eq!(reg.try_get_id(NO_FON_CHAR), Some(NO_FON));
    assert_eq!(reg.try_get_fon(NO_FON), Some(NO_FON_CHAR));
    Ok(())
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
        .filter(|&c| c != NO_FON_CHAR)
        .take(MAX_FON_ID as usize)
    {
        reg.add(c)?;
    }
    assert!(matches!(reg.add('\0'), Err(NoMoreFonIds)));
    Ok(())
}

#[test]
fn test_fonset_empty() {
    let s = FonSet::EMPTY;
    assert!(s.is_empty());
    assert_eq!(s.len(), 0);
    for i in 0..=MAX_FON_ID {
        assert!(!s.contains(i))
    }
}

#[test]
fn test_fonset_real() {
    let mut s = FonSet::new();
    assert!(s.is_real());
    s |= 12;
    assert!(s.is_real());
    s |= NO_FON;
    assert!(!s.is_real());
    s -= 12;
    assert!(!s.is_real());
    s |= 3;
    s |= 101;
    assert!(!s.is_real());
    s -= NO_FON;
    assert!(s.is_real());
}

#[test]
fn test_fonset_default() {
    let s: FonSet = Default::default();
    assert_eq!(s, FonSet::EMPTY);
}

#[test]
fn test_fonset_new() {
    let s: FonSet = FonSet::new();
    assert_eq!(s, FonSet::EMPTY);
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
    let mut s = FonSet::new();
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
    let mut s = FonSet::EMPTY;
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
    let mut s2: FonSet = FonSet::new();
    for i in s1 {
        s2 |= i;
    }
    assert_eq!(s2, s1);
}

#[test]
fn test_fonset_fons() -> Result<()> {
    let mut reg = FonRegistry::new();
    let mut s = FonSet::new();
    let chars = vec!['$', 'q', 'A', 'ç'];
    for &c in chars.iter() {
        s |= reg.add(c)?;
    }
    assert_eq!(s.fons(&reg)?, chars);
    assert!(FonSet::EMPTY.fons(&reg)?.is_empty());
    Ok(())
}

#[test]
fn test_fonsetseq_empty() {
    assert!(FonSet::seq_is_empty(&[]));
    assert!(FonSet::seq_is_empty(&[FonSet::EMPTY]));
    assert!(FonSet::seq_is_empty(&[FonSet::EMPTY, FonSet::EMPTY]));
    assert!(FonSet::seq_is_empty(&[FonSet::EMPTY, FonSet::from(3)]));
    assert!(FonSet::seq_is_empty(&[2.into(), FonSet::EMPTY]));
    assert!(!FonSet::seq_is_empty(&[2.into(), 3.into()]));
}

#[test]
fn test_fonsetseq_real() {
    assert!(FonSet::seq_is_real(&[]));
    assert!(FonSet::seq_is_real(&[FonSet::EMPTY]));
    assert!(!FonSet::seq_is_real(&[FonSet::from(NO_FON)]));
    assert!(!FonSet::seq_is_real(&[
        FonSet::from([2, 3]),
        FonSet::from(NO_FON),
        FonSet::from(80)
    ]));
    assert!(FonSet::seq_is_real(&[
        FonSet::from([2, 3]),
        FonSet::EMPTY,
        FonSet::from(80)
    ]));
}

#[test]
fn test_fonsetseq_valid() {
    assert!(FonSet::seq_is_valid(&[]));
    assert!(FonSet::seq_is_valid(&[FonSet::EMPTY]));
    assert!(FonSet::seq_is_valid(&[FonSet::from(NO_FON)]));
    assert!(!FonSet::seq_is_valid(&[
        FonSet::from([2, 3]),
        FonSet::from(NO_FON),
        FonSet::from(80)
    ]));
    assert!(FonSet::seq_is_valid(&[
        FonSet::from([2, 3]),
        FonSet::EMPTY,
        FonSet::from(80)
    ]));
    assert!(FonSet::seq_is_valid(&[
        FonSet::from(NO_FON),
        FonSet::from([2, 3]),
        FonSet::from(80),
        FonSet::from(NO_FON)
    ]));
}

#[test]
fn test_fonset_first_id() {
    assert_eq!(FonSet::EMPTY.first_id(), None);
    assert_eq!(FonSet::from([17]).first_id(), Some(17));
    assert_eq!(FonSet::from([5, 2, 99]).first_id(), Some(2));
}

#[test]
fn test_fonset_next_id_after() {
    assert_eq!(FonSet::from([17]).next_id_after(17), None);
    assert_eq!(FonSet::from([5, 2, 99]).next_id_after(2), Some(5));
    assert_eq!(FonSet::from([5, 2, 99]).next_id_after(5), Some(99));
    assert_eq!(FonSet::from([5, 2, 99]).next_id_after(99), None);
}

#[test]
fn test_fonset_next_id_after_same_as_iter() {
    for fonset in [
        FonSet::EMPTY,
        FonSet::from([80]),
        FonSet::from([1, 30, 105]),
        FonSet::from(Vec::from_iter(0..MAX_FON_ID).as_slice()),
    ] {
        let by_iter: Vec<FonId> = fonset.iter().collect();
        let mut by_next_id_after = Vec::new();
        let mut current = fonset.first_id();
        while let Some(id) = current {
            by_next_id_after.push(id);
            current = fonset.next_id_after(id);
        }
        assert_eq!(by_next_id_after, by_iter);
    }
}

#[test]
fn test_fonset_seq_for_each_fon_seq() -> Result<()> {
    let mut reg = FonRegistry::new();
    let mut seqs: Vec<Vec<FonId>> = Vec::new();
    FonSet::seq_for_each_fon_seq(&reg.setseq(&["a"])?, |s| seqs.push(s.into()));
    assert_eq!(seqs, vec![reg.seq("a")?]);
    seqs.clear();
    FonSet::seq_for_each_fon_seq(&reg.setseq(&["x", "abc", "bx"])?, |s| seqs.push(s.into()));
    let mut expected = vec![
        reg.seq("xab")?,
        reg.seq("xbb")?,
        reg.seq("xcb")?,
        reg.seq("xax")?,
        reg.seq("xbx")?,
        reg.seq("xcx")?,
    ];
    seqs.sort();
    expected.sort();
    assert_eq!(seqs, expected);
    Ok(())
}

fn collect_matches<M>(matcher: &M, word: &[M::T]) -> Vec<(Vec<M::T>, usize)>
where
    M: MutationRule,
    M::T: Clone,
{
    let mut results: Vec<(Vec<M::T>, usize)> = Vec::new();
    matcher.for_each_match(word, |result, index| results.push((result.clone(), index)));
    results
}

fn collect_matches_at<M>(matcher: &M, index: usize, word: &[M::T]) -> Vec<Vec<M::T>>
where
    M: MutationRule,
    M::T: Clone,
{
    let mut results: Vec<Vec<M::T>> = Vec::new();
    matcher.for_each_match_at(word, index, |result| results.push(result.clone()));
    results
}

#[test]
fn test_fon_set_seq_for_each_match_start() -> Result<()> {
    let mut reg = FonRegistry::new();
    assert_eq!(
        collect_matches(&reg.setseq(&["a"])?, &reg.setseq(&["ax", "bx"])?),
        vec![(reg.setseq(&["a", "bx"])?, 0),]
    );
    Ok(())
}

#[test]
fn test_fon_set_seq_for_each_match_end() -> Result<()> {
    let mut reg = FonRegistry::new();
    assert_eq!(
        collect_matches(&reg.setseq(&["b"])?, &reg.setseq(&["ax", "bx"])?),
        vec![(reg.setseq(&["ax", "b"])?, 1),]
    );
    Ok(())
}

#[test]
fn test_fon_set_seq_for_each_match_middle() -> Result<()> {
    let mut reg = FonRegistry::new();
    assert_eq!(
        collect_matches(&reg.setseq(&["b"])?, &reg.setseq(&["ax", "bx", "cx"])?),
        vec![(reg.setseq(&["ax", "b", "cx"])?, 1),]
    );
    Ok(())
}

#[test]
fn test_fon_set_seq_for_each_match_multiple() -> Result<()> {
    let mut reg = FonRegistry::new();
    assert_eq!(
        collect_matches(
            &reg.setseq(&["ab", "c"])?,
            &reg.setseq(&["r", "a", "cab", "cx", "na"])?
        ),
        vec![
            (reg.setseq(&["r", "a", "c", "cx", "na"])?, 1),
            (reg.setseq(&["r", "a", "ab", "c", "na"])?, 2),
        ]
    );
    Ok(())
}

#[test]
fn test_fon_set_seq_for_each_match_no_match() -> Result<()> {
    let mut reg = FonRegistry::new();
    assert_eq!(
        collect_matches(&reg.setseq(&["n"])?, &reg.setseq(&["a", "b"])?),
        vec![]
    );
    Ok(())
}

#[test]
fn test_fon_set_seq_for_each_match_longer_pattern() -> Result<()> {
    let mut reg = FonRegistry::new();
    assert_eq!(
        collect_matches(&reg.setseq(&["a", "b", "c"])?, &reg.setseq(&["a", "b"])?),
        vec![]
    );
    assert_eq!(
        collect_matches(&reg.setseq(&["a", "b", "c"])?, &reg.setseq(&["b", "c"])?),
        vec![]
    );
    assert_eq!(
        collect_matches(&reg.setseq(&["a", "b", "c"])?, &reg.setseq(&["a"])?),
        vec![]
    );
    Ok(())
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
    assert_eq!(ruleset.get_rule(&['a']), None);
    assert_eq!(ruleset.get_rule(&['b', 'c']), None);
    assert_eq!(ruleset.find_rule(&['d']), None);
    assert_eq!(ruleset.find_rule(&['e', 'f']), None);
}

#[test]
fn test_empty_ruleset_removes_end_anchors() {
    let ruleset = NormalizeRuleSet::new();
    assert_eq!(ruleset.get_rule(&[NO_FON_CHAR]), Some(&[] as &[_]));
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
fn test_normalize_ruleset_and_reg_end_anchored_rule() -> Result<()> {
    let mut ruleset = NormalizeRuleSet::new();
    let mut reg = FonRegistry::new();
    for c in "be".chars() {
        reg.add(c)?;
    }
    ruleset.add_rule(&['s', 't', '_'], &[reg.add('d')?])?;
    assert_eq!((&ruleset, &reg).normalize("best")?, reg.normalize("bed")?);
    Ok(())
}

#[test]
fn test_normalize_ruleset_and_reg_start_anchored_rule() -> Result<()> {
    let mut ruleset = NormalizeRuleSet::new();
    let mut reg = FonRegistry::new();
    for c in "crow".chars() {
        reg.add(c)?;
    }
    ruleset.add_rule(&['_', 'w'], &[reg.add('c')?, reg.add('r')?])?;
    assert_eq!((&ruleset, &reg).normalize("wow")?, reg.normalize("crow")?);
    Ok(())
}

#[test]
fn test_normalize_ruleset_and_reg_empty_result_rule() -> Result<()> {
    let mut ruleset = NormalizeRuleSet::new();
    let mut reg = FonRegistry::new();
    for c in "tests".chars() {
        reg.add(c)?;
    }
    ruleset.add_rule(&['s'], &[])?;
    assert_eq!((&ruleset, &reg).normalize("tests")?, reg.normalize("tet")?);
    Ok(())
}

#[test]
fn test_normalize_ruleset_and_reg_drop_from_start() -> Result<()> {
    let mut ruleset = NormalizeRuleSet::new();
    let mut reg = FonRegistry::new();
    for c in "worddrop".chars() {
        reg.add(c)?;
    }
    ruleset.add_rule(&['_', 'd', 'r', 'o', 'p'], &[])?;
    assert_eq!(
        (&ruleset, &reg).normalize("dropword")?,
        reg.normalize("word")?
    );
    assert_eq!(
        (&ruleset, &reg).normalize("worddrop")?,
        reg.normalize("worddrop")?
    );
    Ok(())
}

#[test]
fn test_normalize_ruleset_and_reg_drop_from_end() -> Result<()> {
    let mut ruleset = NormalizeRuleSet::new();
    let mut reg = FonRegistry::new();
    for c in "worddrop".chars() {
        reg.add(c)?;
    }
    ruleset.add_rule(&['d', 'r', 'o', 'p', '_'], &[])?;
    assert_eq!(
        (&ruleset, &reg).normalize("worddrop")?,
        reg.normalize("word")?
    );
    assert_eq!(
        (&ruleset, &reg).normalize("dropword")?,
        reg.normalize("dropword")?
    );
    Ok(())
}

#[test]
fn test_normalize_ruleset_and_reg_refuses_no_fon_char() -> Result<()> {
    let ruleset = NormalizeRuleSet::new();
    let mut reg = FonRegistry::new();
    for c in "abcjklm".chars() {
        reg.add(c)?;
    }
    assert!(matches!((&ruleset, &reg).normalize("_"), Err(NoSuchFon(c)) if c == '_'));
    assert!(matches!((&ruleset, &reg).normalize("_ab"), Err(NoSuchFon(c)) if c == '_'));
    assert!(matches!((&ruleset, &reg).normalize("c_"), Err(NoSuchFon(c)) if c == '_'));
    assert!(matches!((&ruleset, &reg).normalize("jk_lm"), Err(NoSuchFon(c)) if c == '_'));
    assert!(matches!((&ruleset, &reg).normalize("__"), Err(NoSuchFon(c)) if c == '_'));
    assert!(matches!((&ruleset, &reg).normalize("___"), Err(NoSuchFon(c)) if c == '_'));
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
fn test_buscacfg_search_word_in_dictionary_comes_first() -> Result<()> {
    let word = "word";
    let mut cfg = BuscaCfg::new();
    cfg.load_rules("10: x > y".as_bytes())?;
    for c in word.chars() {
        cfg.fon_registry.add(c)?;
    }
    cfg.add_to_dictionary(word, &cfg.normalize(word)?)?;
    assert_eq!(cfg.search(word)?.flatten().next(), Some((word, 0)));
    Ok(())
}

#[test]
fn test_buscacfg_search_normalize_error() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.load_rules("10: x > y".as_bytes())?;
    assert!(matches!(cfg.search("anything"), Err(NoSuchFon(_))));
    Ok(())
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
    cfg.add_rule(Rule::Alias("Hi", vec![Item::Char('a')]))?;
    assert_eq!(
        cfg.get_alias("Hi")?,
        FonSet::from(cfg.fon_registry.get_id('a')?)
    );
    Ok(())
}

#[test]
fn test_add_rule_alias_with_anchor() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.add_rule(Rule::Alias("Xy", vec![Item::Char('n'), Item::None]))?;
    assert_eq!(
        cfg.get_alias("Xy")?,
        FonSet::from([cfg.fon_registry.get_id('n')?, NO_FON])
    );
    Ok(())
}

#[test]
fn test_resolve_norm_rule_result() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    let r = cfg.resolve_norm_rule_result(&vec![Item::Char('o'), Item::Char('k')])?;
    assert_eq!(r, cfg.normalize("ok")?);
    Ok(())
}

#[test]
fn test_invalid_normalize_rule_rhs() {
    assert!(matches!(
        BuscaCfg::new().load_rules("a b > a _".as_bytes()),
        Err(InvalidNormRule(_))
    ));
    assert!(matches!(
        BuscaCfg::new().load_rules("a b > _ c".as_bytes()),
        Err(InvalidNormRule(_))
    ));
    assert!(matches!(
        BuscaCfg::new().load_rules("C = [z r]\nv > C".as_bytes()),
        Err(InvalidNormRule(_))
    ));
}

#[test]
fn test_invalid_normalize_rule_lhs() {
    assert!(BuscaCfg::new().load_rules(" > a".as_bytes()).is_err());
    assert!(BuscaCfg::new().load_rules("a _ b > c".as_bytes()).is_err());
    assert!(BuscaCfg::new().load_rules("_ _ _ > c".as_bytes()).is_err());
}

#[test]
fn test_resolve_rule_item_set_alias() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    let s = FonSet::from([cfg.fon_registry.add('w')?, cfg.fon_registry.add('p')?]);
    cfg.set_alias("N".into(), s);
    let r = cfg.resolve_rule_item_set(&vec![Item::Alias("N")])?;
    assert_eq!(r, s);
    Ok(())
}

#[test]
fn test_resolve_rule_item_set_alias_plus() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    let s = FonSet::from([cfg.fon_registry.add('w')?, cfg.fon_registry.add('p')?]);
    cfg.set_alias("N".into(), s);
    let r = cfg.resolve_rule_item_set(&vec![Item::Char('q'), Item::Alias("N")])?;
    assert_eq!(r, s | cfg.fon_registry.get_id('q')?);
    Ok(())
}

#[test]
fn test_resolve_rule_item_set_alias_plus_anchor() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    let s = FonSet::from([cfg.fon_registry.add('w')?, cfg.fon_registry.add('p')?]);
    cfg.set_alias("N".into(), s);
    let r = cfg.resolve_rule_item_set(&vec![Item::Alias("N"), Item::None])?;
    assert_eq!(r, s | NO_FON);
    Ok(())
}

#[test]
fn test_add_rule_norm_simple() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.add_rule(Rule::Norm {
        from: vec![vec![Item::Char('a')], vec![Item::Char('b')]],
        to: vec![Item::Char('c'), Item::Char('d')],
    })?;
    assert_eq!(
        cfg.normalize_rules.get_rule(&['a', 'b']),
        Some(cfg.normalize("cd")?.as_slice())
    );
    Ok(())
}

#[test]
fn test_add_rule_norm_anchored_start() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.add_rule(Rule::Norm {
        from: vec![vec![Item::None], vec![Item::Char('x')]],
        to: vec![Item::Char('y')],
    })?;
    assert_eq!(
        cfg.normalize_rules.get_rule(&[NO_FON_CHAR, 'x']),
        Some(cfg.normalize("y")?.as_slice())
    );
    Ok(())
}

#[test]
fn test_add_rule_norm_anchored_end() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.add_rule(Rule::Norm {
        from: vec![vec![Item::Char('e')], vec![Item::None]],
        to: vec![Item::Char('r')],
    })?;
    assert_eq!(
        cfg.normalize_rules.get_rule(&['e', NO_FON_CHAR]),
        Some(cfg.normalize("r")?.as_slice())
    );
    Ok(())
}

#[test]
fn test_add_rule_norm_with_sets() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.add_rule(Rule::Norm {
        from: vec![
            vec![Item::Char('n')],
            vec![Item::Char('a'), Item::Char('o')],
        ],
        to: vec![Item::Char('y')],
    })?;
    assert_eq!(
        cfg.normalize_rules.get_rule(&['n', 'o']),
        Some(cfg.normalize("y")?.as_slice())
    );
    Ok(())
}

#[test]
fn test_add_rule_norm_adds_rhs_only_to_registry() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.add_rule(Rule::Norm {
        from: vec![vec![Item::Char('a')], vec![Item::Char('b')]],
        to: vec![Item::Char('c')],
    })?;
    assert!(cfg.fon_registry.get_id('c').is_ok());
    assert!(cfg.fon_registry.get_id('a').is_err());
    assert!(cfg.fon_registry.get_id('b').is_err());
    Ok(())
}

#[test]
fn test_for_cartesian_product() -> Result<()> {
    let items: Vec<Vec<i32>> = vec![vec![3, 4], vec![7], vec![0, 10, 100]];
    let mut results = Vec::new();
    for_cartesian_product(&items, |i| {
        results.push(Vec::from(i));
        Ok(())
    })?;
    assert_eq!(
        results,
        vec![
            vec![3, 7, 0],
            vec![3, 7, 10],
            vec![3, 7, 100],
            vec![4, 7, 0],
            vec![4, 7, 10],
            vec![4, 7, 100],
        ]
    );
    Ok(())
}

trait TestingFonRegistry {
    fn set(&mut self, chars: &str) -> Result<FonSet>;
    fn seq(&mut self, chars: &str) -> Result<Vec<FonId>>;
    fn setseq(&mut self, sets: &[&str]) -> Result<Vec<FonSet>>;
}

impl TestingFonRegistry for FonRegistry {
    fn set(&mut self, chars: &str) -> Result<FonSet> {
        let mut s = FonSet::new();
        for c in chars.chars() {
            s |= self.add(c)?;
        }
        Ok(s)
    }

    fn seq(&mut self, chars: &str) -> Result<Vec<FonId>> {
        let mut s = Vec::new();
        for c in chars.chars() {
            s.push(self.add(c)?);
        }
        Ok(s)
    }

    fn setseq(&mut self, sets: &[&str]) -> Result<Vec<FonSet>> {
        let mut seq = Vec::new();
        for s in sets {
            seq.push(self.set(s)?);
        }
        Ok(seq)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
struct MockMutationRule<T> {
    matches: Vec<(Vec<T>, usize)>,
    matches_at: Vec<Vec<T>>,
}

impl<T: Clone> MutationRule for MockMutationRule<T> {
    type T = T;

    fn for_each_match_at_using<F: FnMut(&mut Vec<Self::T>)>(
        &self,
        _word: &[Self::T],
        _word_idx: usize,
        mut action: F,
        result_buf: &mut Vec<Self::T>,
    ) {
        for result in &self.matches_at {
            result_buf.clear();
            result_buf.extend_from_slice(result);
            action(result_buf);
        }
    }

    fn for_each_match_using<F: FnMut(&mut Vec<Self::T>, usize)>(
        &self,
        _word: &[Self::T],
        mut action: F,
        result_buf: &mut Vec<Self::T>,
    ) {
        for (result, idx) in &self.matches {
            result_buf.clear();
            result_buf.extend_from_slice(result);
            action(result_buf, *idx);
        }
    }
}

#[derive(Clone, Debug)]
struct ArbReplaceRule<T> {
    remove_idx: usize,
    remove_len: usize,
    replace_with: Vec<T>,
}

impl<T: Arbitrary> Arbitrary for ArbReplaceRule<T> {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        ArbReplaceRule {
            remove_idx: u8::arbitrary(g) as usize,
            remove_len: u8::arbitrary(g) as usize,
            replace_with: Vec::arbitrary(g),
        }
    }
}

fn prep_replace_rule<T: Clone>(
    match_val: Vec<T>,
    match_idx: usize,
    repl_args: ArbReplaceRule<T>,
) -> Option<ReplaceRule<MockMutationRule<T>>> {
    let rule = ReplaceRule {
        matcher: MockMutationRule {
            matches: vec![(match_val.clone(), match_idx)],
            matches_at: vec![match_val.clone()],
        },
        remove_idx: repl_args.remove_idx,
        remove_len: repl_args.remove_len,
        replace_with: repl_args.replace_with.into(),
    };
    if match_idx + rule.get_remove_end() > match_val.len() {
        None
    } else {
        Some(rule)
    }
}

fn match_mock_replace_rule<T: Clone>(
    match_val: Vec<T>,
    match_idx: usize,
    repl_args: ArbReplaceRule<T>,
) -> Option<Vec<T>> {
    prep_replace_rule(match_val, match_idx, repl_args).map(|rule| {
        let (result, _) = collect_matches(&rule, &[]).into_iter().next().unwrap();
        result
    })
}

fn match_at_mock_replace_rule<T: Clone>(
    match_val: Vec<T>,
    match_idx: usize,
    repl_args: ArbReplaceRule<T>,
) -> Option<Vec<T>> {
    prep_replace_rule(match_val, match_idx, repl_args).map(|rule| {
        collect_matches_at(&rule, match_idx, &[])
            .into_iter()
            .next()
            .unwrap()
    })
}

#[quickcheck]
fn replace_rule_for_each_match_inserts(
    match_val: Vec<u8>,
    match_idx: u8,
    repl_args: ArbReplaceRule<u8>,
) -> TestResult {
    let match_idx = match_idx as usize;
    if let Some(result) = match_mock_replace_rule(match_val, match_idx, repl_args.clone()) {
        TestResult::from_bool(
            result[match_idx + repl_args.remove_idx..].starts_with(&repl_args.replace_with),
        )
    } else {
        TestResult::discard()
    }
}

#[quickcheck]
fn replace_rule_for_each_match_at_inserts(
    match_val: Vec<u8>,
    match_idx: u8,
    repl_args: ArbReplaceRule<u8>,
) -> TestResult {
    let match_idx = match_idx as usize;
    if let Some(result) = match_at_mock_replace_rule(match_val, match_idx, repl_args.clone()) {
        TestResult::from_bool(
            result[match_idx + repl_args.remove_idx..].starts_with(&repl_args.replace_with),
        )
    } else {
        TestResult::discard()
    }
}

#[quickcheck]
fn replace_rule_for_each_match_removes(
    match_val: Vec<u8>,
    match_idx: u8,
    repl_args: ArbReplaceRule<u8>,
) -> TestResult {
    let match_idx = match_idx as usize;
    if let Some(result) = match_mock_replace_rule(match_val.clone(), match_idx, repl_args.clone()) {
        TestResult::from_bool(
            result[match_idx + repl_args.remove_idx + repl_args.replace_with.len()..]
                .starts_with(&match_val[match_idx + repl_args.remove_idx + repl_args.remove_len..]),
        )
    } else {
        TestResult::discard()
    }
}

#[quickcheck]
fn replace_rule_for_each_match_at_removes(
    match_val: Vec<u8>,
    match_idx: u8,
    repl_args: ArbReplaceRule<u8>,
) -> TestResult {
    let match_idx = match_idx as usize;
    if let Some(result) =
        match_at_mock_replace_rule(match_val.clone(), match_idx, repl_args.clone())
    {
        TestResult::from_bool(
            result[match_idx + repl_args.remove_idx + repl_args.replace_with.len()..]
                .starts_with(&match_val[match_idx + repl_args.remove_idx + repl_args.remove_len..]),
        )
    } else {
        TestResult::discard()
    }
}

#[quickcheck]
fn test_replace_rule_set_for_each_match(matches: Vec<(Vec<u8>, usize)>) -> bool {
    let rule = MockMutationRule {
        matches,
        ..Default::default()
    };
    collect_matches(&ReplaceRuleSet::from(rule.clone()), &[]) == collect_matches(&rule, &[])
}

#[quickcheck]
fn test_replace_rule_set_for_each_match_at(matches_at: Vec<Vec<u8>>) -> bool {
    let rule = MockMutationRule {
        matches_at,
        ..Default::default()
    };
    collect_matches_at(&ReplaceRuleSet::from(rule.clone()), 0, &[])
        == collect_matches_at(&rule, 0, &[])
}

#[test]
fn test_replace_rule_cost_set_add() -> Result<()> {
    let rule: MockMutationRule<i32> = Default::default();
    let mut rule_set = ReplaceRuleCostSet::new();
    let cost: Cost = 3;
    rule_set.add_rule(rule.clone(), cost);
    let idx = rule_set.costs.iter().position(|&c| c == cost).unwrap();
    assert!(rule_set.rules[idx].rules.contains(&rule));
    Ok(())
}

#[test]
fn test_sliceset_with_fon_set_seq() -> Result<()> {
    let mut set: BTreeSet<Box<[FonSet]>> = BTreeSet::new();
    let mut reg = FonRegistry::new();
    let setseq1 = reg.setseq(&["xy", "z"])?;
    let setseq2 = reg.setseq(&["z", "ay"])?;
    assert!(!set.has_slice(&setseq1));
    assert!(!set.has_slice(&setseq2));
    set.add_slice(&setseq1);
    assert!(set.has_slice(&setseq1));
    assert!(!set.has_slice(&setseq2));
    set.add_slice(&setseq1);
    assert!(set.has_slice(&setseq1));
    assert!(!set.has_slice(&setseq2));
    set.add_slice(&setseq2);
    assert!(set.has_slice(&setseq1));
    assert!(set.has_slice(&setseq2));
    set.add_slice(&setseq1);
    assert!(set.has_slice(&setseq1));
    assert!(set.has_slice(&setseq2));
    Ok(())
}

#[test]
fn test_busca_node_inc_cost() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.load_rules("0: a > b".as_bytes())?;
    cfg.load_rules("12: a > b".as_bytes())?;
    cfg.load_rules("3: a > b".as_bytes())?;
    cfg.load_rules("55: a > b".as_bytes())?;
    let word = cfg.fon_registry.setseq(&["a"])?;
    let mut node = BuscaNode::new(&cfg, &word, 0);
    assert_eq!(node.total_cost, 0);
    assert_eq!(node.inc_cost(&cfg), Some(3));
    assert_eq!(node.inc_cost(&cfg), Some(12));
    assert_eq!(node.inc_cost(&cfg), Some(55));
    assert_eq!(node.inc_cost(&cfg), None);
    Ok(())
}

#[test]
fn test_busca_node_cmp() -> Result<()> {
    let mut cfg = BuscaCfg::new();
    cfg.load_rules("0: a > b".as_bytes())?;
    let word = cfg.fon_registry.setseq(&["a"])?;
    let node1 = BuscaNode::new(&cfg, &word, 0);
    let node2 = BuscaNode::new(&cfg, &word, 78);
    assert!(node1 > node2);
    assert!(node2 < node1);
    assert!(node1 >= node1);
    assert!(node1 <= node1);
    assert!(node2 >= node2);
    assert!(node2 <= node2);
    Ok(())
}
