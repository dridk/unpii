use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PiiCategory {
    Person,
    Email,
    Date,
    Birthdate,
    Location,
    Phone,
    ZipCode,
    Nir,
    Iban,
    Custom(String),
}

impl PiiCategory {
    pub fn placeholder(&self) -> &str {
        match self {
            PiiCategory::Person => "<PERSON>",
            PiiCategory::Email => "<EMAIL>",
            PiiCategory::Date => "<DATE>",
            PiiCategory::Birthdate => "<BIRTHDATE>",
            PiiCategory::Location => "<LOCATION>",
            PiiCategory::Phone => "<PHONE>",
            PiiCategory::ZipCode => "<ZIP_CODE>",
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
            PiiCategory::Phone => 1,
            PiiCategory::Person => 2,
            PiiCategory::Location => 2,
            PiiCategory::ZipCode => 2,
            PiiCategory::Date => 3,
            PiiCategory::Custom(_) => 4,
        }
    }

    pub fn from_label(label: &str) -> PiiCategory {
        match label.to_uppercase().as_str() {
            "PERSON" => PiiCategory::Person,
            "EMAIL" => PiiCategory::Email,
            "DATE" => PiiCategory::Date,
            "BIRTHDATE" => PiiCategory::Birthdate,
            "LOCATION" => PiiCategory::Location,
            "PHONE" => PiiCategory::Phone,
            "ZIP_CODE" => PiiCategory::ZipCode,
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
