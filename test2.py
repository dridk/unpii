import unpii

dataset_regex = {
    "phone": ("tél: 0651565600", "tél: <TELEPHONE>"),
    "phone2": ("tél: 06 51 56 56 00", "tél: <TELEPHONE>"),
    "email": ("email : joe.lafripouille@chu-brest.fr", "email : <EMAIL>"),
    "email_2": ("email : Joe.LaFripouille@chu-brest.fr", "email : <EMAIL>"),
    "email_3": ("bob@gmail.com", "<EMAIL>"),
    "email_4": ("Joe.LaFripouille@chu-****.fr", "<EMAIL>"),
    "nir": ("nir : 164064308898823", "nir : <NIR>"),
    "nir_space": ("nir : 1 64 06 43 088 988 (23)", "nir : <NIR> (23)"),
    "NOM_Prenom": ("name : DUPONT Jean", "name : DUPONT Jean"),
    "Prenom_NOM": ("name : Jean DUPONT", "name : Jean DUPONT"),
    "Nom_compose_Prenom": ("name : De La Fontaine Jean", "name : De La Fontaine Jean"),
    "NOM-NOM_Prenom": ("name : DE-TROIS Jean", "name : DE-TROIS Jean"),
    "NOM_accent_Prénom": ("Monsieur JOÉÇ KAKŸÇ", "Monsieur <NOM>"),
    "NOM_accent_prénom": ("Monsieur JOÉÇ Poçèé", "Monsieur <NOM>"),
    "P._NOM": ("J. Jean", "J. Jean"),
    "Monsieur_NOM_Prenom": ("Monsieur KEAN Jean", "Monsieur <NOM>"),
    "Monsieur_NOM_Prenom_DOUBLE": (
        "Monsieur KEAN KEZAN Jean-Baptiste",
        "Monsieur <NOM>",
    ),
    "INTERNE_NOM-NOM_Prenom": (
        "name : Interne : DE-TROIS Jean",
        "name : Interne : <NOM>",
    ),
    "Titre_Interne": ("Interne", "Interne"),
    "Docteur_NOM_Prenom": ("Docteur DUPONT Jean", "Docteur <NOM>"),
    "Docteur_PRENOM_NOM": ("Docteur DUPONT JEAN", "Docteur <NOM>"),
    "Docteur_newline_PRENOM_NOM": ("Docteur\nDUPONT JEAN", "Docteur\n<NOM>"),
    "Monsieur_P._NOM": ("Monsieur J. Jean", "Monsieur <NOM>"),
    "Monsieur_P._NOM_MAJUSCULE": ("Monsieur J. JEAN", "Monsieur <NOM>"),
    "Monsieur_P._NOM_apostrophe": ("Monsieur J. L'Jean", "Monsieur <NOM>"),
    "Monsieur_P._NOM_apostrophe_MAJ": ("Monsieur J. L'JEAN", "Monsieur <NOM>"),
    "Dr_NOM_Prenom": ("Dr LECLERC Charle", "Dr <NOM>"),
    "Dr_Prenom_NOM": ("Dr Charle LECLERC", "Dr <NOM>"),
    "Dr_P._P._NOM": ("Dr J.F. LECLERC", "Dr <NOM>"),
    "Dr_P._P._Nom": ("Dr J.F. Laclerc", "Dr <NOM>"),
    "Dr_P._P._NOM_2": ("Dr J. LECLERC", "Dr <NOM>"),
    "Professeur_P._de_NOM": ("Professeur L. de LALALAND", "Professeur <NOM>"),
    "Professeur_P._du_NOM": ("Professeur L. du LALALAND", "Professeur <NOM>"),
    "Professeur_P._DE_NOM": ("Professeur L. DE LALALAND", "Professeur <NOM>"),
    "Professeur_P._DU_NOM": ("Professeur L. DU LALALAND", "Professeur <NOM>"),
    "Chef_de_service_P._NOM": ("Chef de service L. LARIDE", "Chef de service <NOM>"),
    "Cheffe_de_service_P._NOM": (
        "Cheffe de service L. LARIDE",
        "Cheffe de service <NOM>",
    ),
    "Mme_P._NOM": ("Mme C. CCCCC", "Mme <NOM>"),
    "DR._NOM": ("DR. LECLERC", "DR. <NOM>"),
    "PR_NAME": ("PR ABGRAL RONAN", "PR <NOM>"),
    "Interne_NOM_Prenom": ("Interne JEAN Jean", "Interne <NOM>"),
    "Externe_NOM_Prenom": ("Externe JEAN Jean", "Externe <NOM>"),
    "nom_phone": (
        "Monsieur JEAN Lasalle, tél : 0647482884",
        "Monsieur <NOM>, tél : <TELEPHONE>",
    ),
    "double_nom": (
        "Monsieur JEAN Jean, Docteur Jeanj JEAN, Madame JEANNE Jean",
        "Monsieur <NOM>, Docteur <NOM>, Madame <NOM>",
    ),
    "test": (
        "Bonjour Monsieur JEAN Jean, voici son numéro : 0606060606 et son email jean.jean@gmail.fr",
        "Bonjour Monsieur <NOM>, voici son numéro : <TELEPHONE> et son email <EMAIL>",
    ),
    "née_madame": ("Madame DUPONT Mariane née MORGAT", "Madame <NOM> née <NOM>"),
    "née_madame_2": ("Nom : DUPONT Mariane née MORGAT", "Nom : <NOM> née <NOM>"),
    "né_monsieur": ("Monsieur J. Jean né LA RUE", "Monsieur <NOM> né <NOM>"),
    "Prof_NOM_PRENOM": ("Professeur JEAN JEAN", "Professeur <NOM>"),
    "Profe_NOM_PRENOM": ("Professeure JEAN JEAN", "Professeure <NOM>"),
    "INT_NOM_PRENOM": ("INT JEAN JEAN", "INT <NOM>"),
    "Date_slash": ("01/12/2000", "<DATE>"),
    "Date_dash": ("01-12-2000", "<DATE>"),
    "Date_incorrect": ("12-22-2000", "12-22-2000"),
    "Date_phrase": ("Brest, le 01/01/2000", "Brest, le <DATE>"),
    "adresse_2": ("155 rue de Brest, 29820, Guilers", "<ADRESSE>"),
    "adresse_3": ("28 RUE DU CHATEAU", "<ADRESSE>"),
    "adresse_4": ("20, bis rue de la Plage", "20, bis <ADRESSE>"),
    "date_mois": ("8 juillet 2020", "<DATE>"),
    "date_fourchette": ("du 15 au 24 octobre 2015", "du 15 au <DATE>"),
    "ville_date": ("BREST, le 4 Juin 2015", "BREST, le <DATE>"),
    "date_dot": ("née le 27.05.31", "née le <DATE>"),
    "zip_name": ("29760 PENMARCH", "<CODE_POSTAL>"),
    "zip_name_2": ("29270 CARHAIX PLOUGUER", "<CODE_POSTAL>"),
    "zip_name_3": ("29270 AIX-EN-PROVENCE", "<CODE_POSTAL>"),
}

passed = 0
failed = 0

for name, (input_text, expected) in dataset_regex.items():
    if input_text is None:
        continue
    result = unpii.mask(input_text)
    status = "PASS" if result == expected else "FAIL"
    if status == "FAIL":
        failed += 1
        print(f"FAIL  {name}")
        print(f"  input:    {repr(input_text)}")
        print(f"  expected: {repr(expected)}")
        print(f"  got:      {repr(result)}")
        print()
    else:
        passed += 1

print(f"\n{'='*50}")
print(f"PASS: {passed}  FAIL: {failed}  TOTAL: {passed + failed}")
