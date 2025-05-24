/*
 * Copyright (c) 2025. Hunc codicem scripsit Lajos, qui dicitur Kovács, ad suum solatium et eruditionem.
 */

package laj.generators;

import laj.generators.utils.GeneratorParams;
import java.util.Random;


import java.util.Arrays;


/**
 * CPU‐oldali véletlenszám‐generátor Robert Floyd mintavételezéssel,
 * amely blokk‐méretekben (k) húz ismétlés nélküli, növekvő sorozatú értékeket
 * [min…max] tartományból, és ezeket fűzi fel az N hosszú vektorra.
 */
public class DrawJavaRandomGenerator extends CpuRandomGenerator{
    private final Random random;
    private final int k;
    private final int min;


    public DrawJavaRandomGenerator(GeneratorParams params) {
        super(params);
        this.k      = params.k();
        this.min    = params.min();
        this.random = new Random(params.seed());
    }

    @Override
    public void generate(long nn) {
        int n = (int)nn;
        int blocks = (n + k - 1) / k;  // hány blokk szükséges

        for (int b = 0; b < blocks; b++) {
            int offset = b * k;
            int blockSize = Math.min(k, n - offset);

            // Robert Floyd mintavételezés erre a blokkra
            int[] S = new int[blockSize];
            int idx = 0;
            for (int i = range - blockSize + 1; i <= range; i++) {
                int t = random.nextInt(i);  // [0..i-1]
                boolean found = false;
                for (int j = 0; j < idx; j++) {
                    if (S[j] == t) { found = true; break; }
                }
                S[idx++] = found ? (i - 1) : t;
            }

            // Offset hozzáadása és növekvő sorrend
            for (int i = 0; i < blockSize; i++) {
                S[i] += min;
            }
            Arrays.sort(S);

            // Blokk kiírása a vektor megfelelő szakaszába
            for (int i = 0; i < blockSize; i++) {
                vector[offset + i] = S[i];
            }
        }
    }

}
