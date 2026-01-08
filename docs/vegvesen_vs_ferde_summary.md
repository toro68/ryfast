# Sammenligning: Vegvesen-tellinger vs Ferde-bompasseringer (Ryfast)

Datagrunnlag:
- Ferde: `docs/2025_2024 trafikk og inntekt_Ryfast (1).xlsx` (kolonnene **Passeringer** og **Fritakspasseringer**)
- Vegvesen: `trafikkdata-api.atlas.vegvesen.no` summering av målepunkter for **Ryfast (sum tunneler, inkl pårampe)** (`99040V2725982`, `00911V2725983`, `10239V2725979`, `62464V2725991`, `92743V2726085`, `25926V2725990`)
- Beregning Vegvesen: (månedsvis gjennomsnittlig døgntrafikk) × (antall dager i måneden), summert over målepunkter

Resultat (årssum):
- 2024: Vegvesen `7 162 124` vs Ferde `6 940 889` (ratio `1,0319`)
- 2025: Vegvesen `7 233 497` vs Ferde `7 197 420` (ratio `1,0050`)*

*Viktig for 2025: To av Hundvåg-målepunktene mangler **august** i Vegvesen-APIet, så Vegvesen-tallet for august er basert på 4/6 målepunkter. Hvis vi sammenligner kun måneder der alle 6 punkter har data (11 måneder), er ratio `1,0373` (Vegvesen høyere enn Ferde).

Outputfil med månedstall:
- `docs/vegvesen_vs_ferde_ryfast_2024_2025.csv`

Merk:
- Vegvesen-tellingene er trafikktellinger i målepunkter, mens Ferde er bompasseringer/transaksjoner. Tallene trenger derfor ikke være 1:1.
