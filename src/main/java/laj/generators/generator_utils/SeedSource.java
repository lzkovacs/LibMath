/*
 * Copyright (c) 2024-2025. Hunc codicem scripsit Lajos, qui dicitur Kovács, ad suum solatium et eruditionem.
 */

package laj.generators.generator_utils;
/**
 * Felsoroló típus, amely különböző véletlenszám-forrásokat reprezentál, melyek maggeneráláshoz használhatók fel.
 * SYSTEM: A rendszer véletlenszám-generátorát használja.
 * RANDOM_ORG: A RANDOM.ORG szolgáltatást használja véletlenszám-generáláshoz.
 * ANU_QUANTUM: Az ANU Quantum Random Number Algorithm szolgáltatást veszi igénybe.
 * FOURMILAB: A Fourmilab HotBits szolgáltatást alkalmazza véletlenszámok előállításához.
 * NANOTIME: A rendszer nanoszekundum időbélyegét használja véletlenszám-generáláshoz.
 */
public enum SeedSource {
    SYSTEM,
    RANDOM_ORG,
    ANU_QUANTUM,
    FOURMILAB,
    NANOTIME
}
