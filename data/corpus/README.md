# Tourism Corpus (Moroccan focus)

Drop your source documents in this directory. Supported extensions:

- `.pdf`  — official guides, ministry publications, regional brochures
- `.txt`  — plain text exports
- `.md`   — markdown notes

The ingestion script (`python -m src.ingest`) walks this folder recursively,
so feel free to organize content in subfolders (e.g., `regions/`, `culture/`,
`practical/`).

## Suggested sources for a strong corpus (10-30 documents is plenty)

1. **ONMT (Office National Marocain du Tourisme)** publications
   - https://www.visitmorocco.com/  — regional guides, brochures (FR/EN)
2. **UNESCO World Heritage Sites in Morocco** (Volubilis, Fes Medina,
   Marrakech Medina, Ait Ben Haddou, Essaouira, Tetouan, Meknes, Rabat).
   - UNESCO whc.unesco.org PDF descriptions
3. **Ministry of Tourism / National Tourism Strategy** white papers
   (e.g., "Vision 2020", "Feuille de route 2023-2026")
4. **Regional Tourism Councils (CRT)** — Marrakech-Safi, Fes-Meknes,
   Souss-Massa, Tanger-Tetouan-Al Hoceima, Drâa-Tafilalet, etc.
5. **Lonely Planet / Routard / Petit Fute** chapter samples
6. **Wikivoyage** Morocco region articles (export as PDF)
7. **Visa / customs / practical info** from consulates
8. **Gastronomy & culture** — Maroc Cuisine pages, IRCAM (Amazigh culture)

## Size & quality guidelines

- Keep total corpus under ~200 MB to stay laptop-friendly.
- Prefer text-extractable PDFs (avoid scanned-only PDFs unless you add OCR).
- Mix languages if you want — the embedding model is multilingual.

## After dropping files here

```bash
python -m src.ingest --reset   # build a fresh Chroma index
```
