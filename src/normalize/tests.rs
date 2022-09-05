use super::*;

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
    let norm1 = &[Fon::from(8u8), Fon::from(2u8)] as &[_];
    let norm2 = &[Fon::from(18u8), Fon::from(7u8), Fon::from(91u8)] as &[_];
    let mut ruleset = NormalizeRuleSet::new();
    ruleset.add_rule(&['c'], norm1)?;
    assert_eq!(ruleset.longest_rule(), 1);
    ruleset.add_rule(&['a', 'b'], norm2)?;
    assert_eq!(ruleset.longest_rule(), 2);
    assert_eq!(ruleset.get_rule(&['a']), None);
    assert_eq!(ruleset.get_rule(&['a', 'b']), Some(norm2));
    assert_eq!(ruleset.get_rule(&['c']), Some(norm1));
    assert_eq!(ruleset.get_rule(&['x', 'c']), None);
    assert_eq!(ruleset.get_rule(&['c', 'x']), None);
    assert_eq!(ruleset.get_rule(&['d']), None);
    Ok(())
}

#[test]
fn test_ruleset_add_duplicate() -> Result<()> {
    let norm1 = &[Fon::from(8u8), Fon::from(2u8)];
    let norm2 = &[Fon::from(18u8), Fon::from(7u8), Fon::from(91u8)];
    let norm3 = &[Fon::from(10u8)];
    let norm4 = &[Fon::from(100u8)];
    let mut ruleset = NormalizeRuleSet::new();
    ruleset.add_rule(&['c'], norm1)?;
    assert!(matches!(
        ruleset.add_rule(&['c'], norm2),
        Err(DuplicateNormRule(_))
    ));
    ruleset.add_rule(&['a', 'b'], norm3)?;
    assert!(matches!(
        ruleset.add_rule(&['c'], norm2),
        Err(DuplicateNormRule(_))
    ));
    assert!(matches!(
        ruleset.add_rule(&['a', 'b'], norm4),
        Err(DuplicateNormRule(_))
    ));
    Ok(())
}

#[test]
fn test_ruleset_find() -> Result<()> {
    let norm1 = &[Fon::from(8u8), Fon::from(2u8)] as &[_];
    let norm2 = &[Fon::from(18u8), Fon::from(7u8), Fon::from(91u8)] as &[_];
    let mut ruleset = NormalizeRuleSet::new();
    ruleset.add_rule(&['a', 'b'], norm1)?;
    ruleset.add_rule(&['c'], norm2)?;
    assert_eq!(ruleset.find_rule(&['x']), None);
    assert_eq!(ruleset.find_rule(&['x', 'a', 'b', 'y']), None);
    assert_eq!(ruleset.find_rule(&['x', 'c', 'y']), None);
    assert_eq!(ruleset.find_rule(&['c']), Some((1, norm2)));
    assert_eq!(ruleset.find_rule(&['c', 'z']), Some((1, norm2)));
    assert_eq!(ruleset.find_rule(&['a', 'b']), Some((2, norm1)));
    assert_eq!(ruleset.find_rule(&['a', 'b', 'n']), Some((2, norm1)));
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
