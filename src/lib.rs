use thiserror::Error;

pub type Result<T> = std::result::Result<T, FonError>;

#[derive(Error, Clone, PartialEq, Eq, Debug)]
pub enum FonError {
    #[error("No such fon {0:?}")]
    NoSuchFon(char),
    #[error("Ran out of fon ids")]
    NoMoreFonIds,
    #[error("No such fon id {0}")]
    NoSuchFonId(FonId),
}

use FonError::*;

const MAX_FON_ID: FonId = 127;

type FonId = usize;

#[derive(Clone, Debug)]
pub struct FonRegistry {
    fons: Vec<char>,
}

impl FonRegistry {
    pub fn new() -> FonRegistry {
        FonRegistry { fons: Vec::new() }
    }

    pub fn add(&mut self, fon: char) -> Result<FonId> {
        match self.try_get_id(fon) {
            Some(id) => Ok(id),
            None => {
                let id = self.fons.len() as FonId;
                if id <= MAX_FON_ID {
                    self.fons.push(fon);
                    Ok(id)
                } else {
                    Err(NoMoreFonIds)
                }
            }
        }
    }

    pub fn try_get_id(&self, fon: char) -> Option<FonId> {
        self.fons.iter().position(|&f| f == fon)
    }

    pub fn get_id(&self, fon: char) -> Result<FonId> {
        self.try_get_id(fon).ok_or(NoSuchFon(fon))
    }

    pub fn try_get_fon(&self, id: FonId) -> Option<char> {
        self.fons.get(id).cloned()
    }

    pub fn get_fon(&self, id: FonId) -> Result<char> {
        self.try_get_fon(id).ok_or(NoSuchFonId(id))
    }
}

#[test]
fn test_fon_registry_empty() {
    let reg = FonRegistry::new();
    for c in ['q', 'é', '\0'] {
        assert_eq!(reg.get_id(c), Err(NoSuchFon(c)));
    }
    for i in [0, 1, 2, 35, MAX_FON_ID] {
        assert_eq!(reg.get_fon(i), Err(NoSuchFonId(i)));
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
    for c in (1..1000000u32)
        .flat_map(char::from_u32)
        .take((MAX_FON_ID as usize) + 1)
    {
        reg.add(c)?;
    }
    assert_eq!(reg.add('\0'), Err(NoMoreFonIds));
    Ok(())
}
