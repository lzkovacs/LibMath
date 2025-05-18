/*
 * Copyright (c) 2024-2025. Hunc codicem scripsit Lajos, qui dicitur Kovács, ad suum solatium et eruditionem.
 */
package laj.kernels;

import jcuda.Pointer;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdeviceptr;
import laj.kernels.kernel_util.CudaKernelBase;
import laj.kernels.kernel_util.GridOptimizer;

import static jcuda.driver.JCudaDriver.*;

/**
 * Transzformációs kernel wrapper: eltolja és skálázza a véletlenszám-vektort.
 * A kernel "transformKernelVec4" néven van definiálva a resource PTX fájlban.
 */
public class ScaleKernel extends CudaKernelBase {

    private final GridOptimizer gridOpt;

    /**
     * Konstruktor: örökli a külső kontextust és betölti a PTX modult.
     * Lekérdezi a GPU-korlátokat a GridOptimizer segítségével.
     *
     * @param context A megosztott CUDA kontextus
     */
    public ScaleKernel(CUcontext context) {
        super(context);
        this.gridOpt = new GridOptimizer();
    }

    @Override
    protected String getKernelFileName() {
        return "kernels/scale_kernel.cu";
    }

    @Override
    protected String getKernelFunctionName() {
        return "scaleKernelVec4";
    }

    /**
     * A vektor értékeinek áttranszformálása: y[i] = x[i] * diff + base
     * GridOptimizer-rel választja ki a grid és block méreteket.
     *
     * @param xDevice GPU pointer bemeneti adatokra
     * @param yDevice GPU pointer kimeneti adatokra
     * @param minVal alsó határ (base)
     * @param maxVal felső határ (százalékos diff)
     * @param N elemszám a vektorban
     */
    public void scale(CUdeviceptr xDevice,
                      CUdeviceptr yDevice,
                      int minVal,
                      int maxVal,
                      int N) {
        // diff és base kiszámítása
        float diff = (float)(maxVal - minVal);
        float base = (float)minVal;

        // Grid és block méretek optimalizálása
        int[] dims = gridOpt.optimize(N);
        int gridX = dims[0];
        int gridY = dims[1];
        int blockSize = dims[2];

        // Kernel paraméterek összeállítása
        Pointer kernelParams = Pointer.to(
                Pointer.to(xDevice),
                Pointer.to(yDevice),
                Pointer.to(new float[]{diff}),
                Pointer.to(new float[]{base}),
                Pointer.to(new int[]{N})
        );

        // Kernel indítása
        cuCtxSetCurrent(context);
        cuLaunchKernel(
                function,
                gridX, gridY, 1,
                blockSize, 1, 1,
                0,
                null,
                kernelParams,
                null
        );

        // Szinkronizálás a kernel befejezéséhez
        cuCtxSetCurrent(context);
        cuCtxSynchronize();
    }
}