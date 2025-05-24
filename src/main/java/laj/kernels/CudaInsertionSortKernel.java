/*
 * Copyright (c) 2025. Hunc codicem scripsit Lajos, qui dicitur Kovács, ad suum solatium et eruditionem.
 */

package laj.kernels;

import jcuda.Pointer;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUresult;
import laj.kernels.utils.CudaKernelBase;
import laj.kernels.utils.GridOptimizer;

import static jcuda.driver.JCudaDriver.*;

/**
 * Kernelfuttató osztály: insertion sort algoritmus a GPU-n.
 * <p>
 * A GridOptimizer-rel választja ki a megfelelő grid és block méreteket.
 * </p>
 */
public class CudaInsertionSortKernel extends CudaKernelBase {

    private final GridOptimizer gridOpt;

    /**
     * Konstruktor: örökli a megosztott CUDA kontextust és inicializálja a GridOptimizert.
     * @param context A közös CUcontext
     */
    public CudaInsertionSortKernel(CUcontext context) {
        super(context);
        this.gridOpt = new GridOptimizer();
    }

    @Override
    protected String getKernelFileName() {
        return "kernels/insertionsort_kernel.cu";
    }

    @Override
    protected String getKernelFunctionName() {
        return "sortKernel";
    }

    /**
     * A device memóriában lévő vektor rendezése insertion sort kernel segítségével.
     * @param deviceVector A rendezendő vektor eszközpointere
     * @param N A vektor elemeinek száma
     * @param j További kernel-paraméter (pl. lépésindex)
     */
    public void sort(CUdeviceptr deviceVector, int N, int j) {
        // Optimalizált grid és block méretek meghatározása
        int[] dims = gridOpt.optimize(N);
        int gridX = dims[0];
        int gridY = dims[1];
        int blockSize = dims[2];

        // Kernel paraméterek összeszerelése
        Pointer kernelParams = Pointer.to(
                Pointer.to(deviceVector),
                Pointer.to(new int[]{N}),
                Pointer.to(new int[]{j})
        );

        // Kernel elindítása
        cuCtxSetCurrent(context);
        int result = cuLaunchKernel(
                function,
                gridX, gridY, 1,
                blockSize, 1, 1,
                0,
                null,
                kernelParams,
                null
        );
        // Beállítjuk az aktuális kontextust a szinkronizálás előtt
        cuCtxSetCurrent(context);
        cuCtxSynchronize();

        if (result != CUresult.CUDA_SUCCESS) {
            throw new RuntimeException("Kernel indítása sikertelen: " + result);
        }

    }
}
