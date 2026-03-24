use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PiiCategory {
    Nom,
    Email,
    Date,
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

    pub fn from_label(label: &str) -> PiiCategory {
        match label.to_uppercase().as_str() {
            "NOM" => PiiCategory::Nom,
            "EMAIL" => PiiCategory::Email,
            "DATE" => PiiCategory::Date,
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
