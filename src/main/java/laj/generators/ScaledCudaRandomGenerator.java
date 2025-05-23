/*
 * Copyright (c) 2025. Hunc codicem scripsit Lajos, qui dicitur Kovács, ad suum solatium et eruditionem.
 */
package laj.generators;

import jcuda.driver.CUcontext;
import laj.generators.utils.Algorithm;
import laj.generators.utils.GeneratorParams;
import laj.kernels.ScaleKernel;

/**
 * <h2>ScaledCudaRandomGenerator</h2>
 * <p>
 * Olyan GPU-alapú véletlenszám-generátor, amely először uniform eloszlású számokat generál,
 * majd azokat egy adott [min, max] intervallumba skálázza CUDA kernellel.
 * </p>
 * <p>
 * Használata tipikusan akkor javasolt, ha a GPU erőforrásait kívánjuk kihasználni nagyméretű 
 * vagy intenzív véletlenszám-generálási feladatokhoz.
 * </p>
 * 
 * <p><b>Példa használat:</b></p>
 * <pre>
 * {@code
 * GeneratorParams params = ...;
 * CUcontext context = ...;
 * try (ScaledCudaRandomGenerator gen = new ScaledCudaRandomGenerator(params, context)) {
 *     gen.generate(1000);
 *     CUdeviceptr devPtr = gen.getDevPtr();
 *     // Feldolgozás...
 * }
 * }
 * </pre>
 * @see UniformCudaRandomGenerator
 * @see ScaleKernel
 */
public class ScaledCudaRandomGenerator extends GpuRandomGenerator {
    /**
     * Uniform eloszlású GPU generátor példány.
     * <p>
     * A generált vektort ez az egység tölti fel véletlenszámokkal,
     * mielőtt azok skálázásra kerülnének.
     */
    private final UniformCudaRandomGenerator uniformGen;

    /**
     * CUDA kernel, amely a véletlen számokat min és max közé skálázza.
     * <p>
     * A generált uniform vektor értékeit alakítja át a kívánt tartományba.
     */
    private final ScaleKernel scaleKernel;

    /**
     * Létrehoz egy új ScaledCudaRandomGenerator példányt a megadott paraméterekkel és CUDA kontextussal.
     * <ul>
     *     <li>Beállítja az uniform random generátor paramétereit.</li>
     *     <li>Előkészíti a skálázó kernel erőforrásokat.</li>
     * </ul>
     *
     * @param params Véletlengenerátor konfiguráció (min, max, algoritmus, stb.).
     * @param context A használandó CUDA kontextus, vagy {@code null}, ha új kontextust kell létrehozni.
     */
    public ScaledCudaRandomGenerator(GeneratorParams params, CUcontext context) {
        super(params, context);
        // Az uniform algoritmus beállítása (például SCALE_XORWOW → XORWOW)
        Algorithm uniformAlg = params.getBaseAlgorithm();
        GeneratorParams uniformParams = params
                .withAlgorithm(uniformAlg)
                .withTransformEnabled(true);
        // Uniform generátor létrehozása
        this.uniformGen = new UniformCudaRandomGenerator(params, context);
        // Skálázó kernel inicializálása
        this.scaleKernel = new ScaleKernel(context);
    }

    /**
     * Véletlenszámok generálása és skálázása a GPU-n.
     * <br>
     * Először az {@link #uniformGen} tölti fel a vektort, majd a {@link ScaleKernel}
     * azokat a kívánt [min, max] tartományba skálázza.
     *
     * @param n A generálandó véletlenszámok száma.
     */
    @Override
    public void generate(long n) {
        uniformGen.generate(n);
        scaleKernel.scale(
                uniformGen.getDevPtr(),
                devPtr,
                params.min(),
                params.max(),
                params.vectorSize()
        );
    }


    /**
     * Az erőforrások felszabadítása.
     * <br>
     * Lezárja a használt CUDA kernel és uniform generátor erőforrásait.
     */
    @Override
    public void close() {
        scaleKernel.close();
        uniformGen.close();
    }

}