
/*
 * Copyright (c) 2025. Hunc codicem scripsit Lajos, qui dicitur Kovács, ad suum solatium et eruditionem.
 */
package laj.generators.utils;


/**
 * Paraméter-objektum a generátorok konfigurációjához.
 *
 * @param algorithm A véletlenszám-generáláshoz használt algoritmus (alap vagy wrapper)
 * @param seed A generátor seed értéke
 * @param vectorSize A generált vektor hossza
 * @param transformEnabled Jelzi, hogy alkalmazzuk-e a transzformációs lépést
 * @param min A transformáció során használt minimális érték
 * @param max A transformáció során használt maximális érték
 * @param k A rendezési blokkméret (DRAW/SORTED algoritmusoknál mintavételezéshez)
 */
public record GeneratorParams(
        Algorithm algorithm,
        long seed,
        int vectorSize,
        boolean transformEnabled,
        int min,
        int max,
        int k
) {
    /**
     * Létrehoz egy új példányt az adott algoritmussal, megtartva a többi beállítást.
     * @param algorithm A módosított algoritmus
     * @return Új GeneratorParams objektum
     */
    public GeneratorParams withAlgorithm(Algorithm algorithm) {
        return new GeneratorParams(algorithm, seed, vectorSize, transformEnabled, min, max, k);
    }

    /**
     * Létrehoz egy új példányt a transformEnabled kapcsoló módosításával.
     * @param transformEnabled Az új érték
     * @return Új GeneratorParams objektum
     */
    public GeneratorParams withTransformEnabled(boolean transformEnabled) {
        return new GeneratorParams(algorithm, seed, vectorSize, transformEnabled, min, max, k);
    }

    /**
     * Létrehoz egy új példányt a blokkméret (k) módosításával.
     * @param k Az új blokkméret
     * @return Új GeneratorParams objektum
     */
    public GeneratorParams withBlockSize(int k) {
        return new GeneratorParams(algorithm, seed, vectorSize, transformEnabled, min, max, k);
    }

    /**
     * Kényelmi metódus: az aktuális algoritmus tényleges alapszintű (wrapper nélküli) értéke.
     * @return A base algoritmus, ha wrapper, egyébként önmaga
     */
    public Algorithm getBaseAlgorithm() {
        return algorithm.getBase();
    }
}