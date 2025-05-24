/*
 * Copyright (c) 2025. Hunc codicem scripsit Lajos, qui dicitur Kovács, ad suum solatium et eruditionem.
 */

package laj.generators;


import laj.generators.utils.GeneratorParams;
import java.util.Random;


public class UniformJavaRandomGenerator extends CpuRandomGenerator {
    private final Random random;
    private final int min;

    public UniformJavaRandomGenerator(GeneratorParams params) {
        super(params);
        this.min = params.min();
        this.random = new Random(params.seed());
    }

    @Override
    public void generate(long n) {

        for (int i = 0; i < n; i++) {
            vector[i] = random.nextInt(range) + min;
        }
    }

}
