import timeit
import unpii

with open("test.md") as f:
    text = f.read()

print(f"Taille du texte : {len(text)} caractères, {len(text.splitlines())} lignes")
print()

# Anonymiser une fois pour voir le résultat
result = unpii.mask(text, mode="paranoid")
print("=== Résultat anonymisé ===")
print(result)
print()

# Benchmark
n = 1000
t = timeit.timeit(lambda: unpii.mask(text), number=n)
print(f"=== Benchmark ===")
print(f"{n} itérations en {t:.3f}s")
print(f"{t/n*1000:.3f} ms par appel")
print(f"{n/t:.0f} documents/seconde")
