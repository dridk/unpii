import time

import polars as pl
import unpii

# Textes médicaux variés pour simuler des comptes rendus
SAMPLES = [
    "Dr Martin a examiné le patient au 06 12 34 56 78, email: martin@chu-brest.fr",
    "Mme Dupont née le 22/09/1992, habitant 12 rue de la Paix à Paris",
    "Le patient présente une maladie de Parkinson stade 3, NIR 1 92 03 75 108 042 38",
    "Consultation du 15/01/2024 avec Dr Lefebvre, IBAN FR76 3000 6000 0112 3456 7890 189",
    "M. Bernard, né le 03/04/1965, demeurant 45 boulevard Victor Hugo 69003 Lyon",
    "Antécédents: Alzheimer diagnostiqué en 2019, suivi par Dr Moreau au 01 45 67 89 10",
    "Compte rendu opératoire - Patiente Durand, opérée le 10/03/2024 pour fracture du col",
    "Contact: sophie.lambert@hopital-nantes.fr, tél: 02 40 08 33 44",
    "Transfert vers CHU Bordeaux, Dr Petit, le 28/02/2024, patient Robert Martin",
    "Prescription: M. Leroy Jean-Pierre, 78 avenue des Champs 75008 Paris, né le 11/11/1950",
]

N = 1_000_000
texts = [SAMPLES[i % len(SAMPLES)] for i in range(N)]

df = pl.DataFrame({"text": texts})

unpii.set_max_threads(20)

print(f"DataFrame: {df.shape[0]} lignes")
print(f"Threads: {unpii.get_max_threads()}")
print()

start = time.perf_counter()
df_masked = unpii.anonymize_dataframe(df, "text")
elapsed = time.perf_counter() - start

print(df_masked.head(10))
print(f"\n{N} lignes anonymisées en {elapsed:.3f}s ({N / elapsed:.0f} lignes/s)")
