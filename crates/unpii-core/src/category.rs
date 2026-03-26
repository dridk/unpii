use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PiiCategory {
    Nom,
    Email,
    Date,
    Birthdate,
    Adresse,
    Telephone,
    CodePostal,
    Nir,
    Iban,
    Custom(String),
}

impl PiiCategory {
    pub fn placeholder(&self) -> &str {
        match self {
            PiiCategory::Nom => "<NOM>",
            PiiCategory::Email => "<EMAIL>",
            PiiCategory::Date => "<DATE>",
            PiiCategory::Birthdate => "<BIRTHDATE>",
            PiiCategory::Adresse => "<ADRESSE>",
            PiiCategory::Telephone => "<TELEPHONE>",
            PiiCategory::CodePostal => "<CODE_POSTAL>",
            PiiCategory::Nir => "<NIR>",
            PiiCategory::Iban => "<IBAN>",
            PiiCategory::Custom(s) => {
                // We leak a string here but it's only called for static labels
                // In practice, custom categories are few and long-lived
                Box::leak(format!("<{}>", s).into_boxed_str())
            }
        }
    }

    /// Priority for overlap resolution: lower = higher priority.
    /// More specific categories (BIRTHDATE) beat generic ones (DATE).
    pub fn priority(&self) -> u8 {
        match self {
            PiiCategory::Birthdate => 0,
            PiiCategory::Nir => 1,
            PiiCategory::Iban => 1,
            PiiCategory::Email => 1,
            PiiCategory::Telephone => 1,
            PiiCategory::Nom => 2,
            PiiCategory::Adresse => 2,
            PiiCategory::CodePostal => 2,
            PiiCategory::Date => 3,
            PiiCategory::Custom(_) => 4,
        }
    }

    pub fn from_label(label: &str) -> PiiCategory {
        match label.to_uppercase().as_str() {
            "NOM" => PiiCategory::Nom,
            "EMAIL" => PiiCategory::Email,
            "DATE" => PiiCategory::Date,
            "BIRTHDATE" => PiiCategory::Birthdate,
            "ADRESSE" => PiiCategory::Adresse,
            "TELEPHONE" => PiiCategory::Telephone,
            "CODE_POSTAL" => PiiCategory::CodePostal,
            "NIR" => PiiCategory::Nir,
            "IBAN" => PiiCategory::Iban,
            other => PiiCategory::Custom(other.to_string()),
        }
    }
}

impl fmt::Display for PiiCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.placeholder())
    }
}
