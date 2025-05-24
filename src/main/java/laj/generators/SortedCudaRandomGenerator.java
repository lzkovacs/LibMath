/*
 * Copyright (c) 2025. Hunc codicem scripsit Lajos, qui dicitur Kovács, ad suum solatium et eruditionem.
 */
package laj.generators;

import laj.generators.utils.GeneratorParams;
import jcuda.driver.CUcontext;
import laj.kernels.CudaInsertionSortKernel;
import laj.kernels.ScaleKernel;

/**
 * Rendezett GPU-generátor:
 * uniform véletlenszámok generálása, skálázás [min,max], majd rendezés.
 */
public class SortedCudaRandomGenerator extends GpuRandomGenerator {
    private final UniformCudaRandomGenerator uniformGen;
    private final ScaleKernel scaleKernel;
    private final CudaInsertionSortKernel sortKernel;
    private final int k;


    /**
     * Konstruktor: beállítja a belső uniform generátort, skálázó és rendező kernelt a megadott kontextussal.
     * @param params paraméterek (algoritmus, seed, vektorméret, min, max, k)
     * @param context A használandó CUDA kontextus, vagy null, ha új kontextust kell létrehozni
     */
    public SortedCudaRandomGenerator(GeneratorParams params, CUcontext context) {
        super(params, context);

        // belső uniform generátor a base algoritmus alapján
        uniformGen = new UniformCudaRandomGenerator(
                params.withAlgorithm(params.getBaseAlgorithm())
                        .withTransformEnabled(true),
                context
        );

        // skálázó kernel shared memóriában
        scaleKernel = new ScaleKernel(context);

        // rendező kernel insertion sort-tal
        sortKernel  = new CudaInsertionSortKernel(context);

        this.k = params.k();
    }

    /**
     * Generálás:
     * 1) uniform véletlenszámok
     * 2) skálázás [min,max]
     * 3) rendezés insertion sort-tal
     */
    @Override
    public void generate(long n) {
        uniformGen.generate(n);

        scaleKernel.scale(
                uniformGen.getDevPtr(), devPtr,
                params.min(), params.max(), params.vectorSize()
        );

        sortKernel.sort(
                devPtr, params.vectorSize(), k
        );
    }


    /**
     * Erőforrások felszabadítása: generátor, skálázó és rendező zárása.
     */
    @Override
    public void close() {
        sortKernel.close();
        scaleKernel.close();
        uniformGen.close();
    }

}
