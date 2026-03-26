import unpii
import polars as pl
import readline

while True:

    text = input("text: ")
    print(unpii.anonymize(text, mode="paranoid"))

# print("\n=== Masquage étoiles ===")
# print(unpii.mask(text, mask="stars"))

# print("\n=== Mode paranoid ===")
# print(unpii.mask(text, mode="paranoid"))

# print("\n=== Ignore TELEPHONE ===")
# print(unpii.mask(text, ignore_groups=["TELEPHONE"]))

# print("\n=== find_spans ===")
# for span in unpii.find_spans(text):
#     print(f"  {span} -> '{text[span.start:span.end]}'")

# print("\n=== Polars ===")
# df = pl.DataFrame({"text": [
#     "Dr Martin au 06 12 34 56 78",
#     "Email: joe@chu-brest.fr",
#     "Maladie de Parkinson",
#     "Mme Dupont née le 22/09/1992",
# ]})
# print(df.with_columns(pl.col("text").unpii.mask()))
