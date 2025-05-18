/*
 * Copyright (c) 2024-2025. Hunc codicem scripsit Lajos, qui dicitur Kovács, ad suum solatium et eruditionem.
 */

package laj.kernels.kernel_util;

import jcuda.driver.CUdevice;
import jcuda.driver.CUdevice_attribute;
import jcuda.driver.JCudaDriver;
import lombok.extern.log4j.Log4j2;

import static jcuda.driver.JCudaDriver.*;

/**
 * Segédosztály a CUDA kernel grid és block méretek optimalizálásához.
 */
@Log4j2
public class GridOptimizer {
    private final int maxGridX;
    private final int maxGridY;
    private final int maxThreadsPerBlock;

    /**
     * Inicializálja a GridOptimizer-t a GPU korlátok lekérdezésével.
     */
    public GridOptimizer() {
        log.debug("GridOptimizer inicializálása");
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);

        int[] gridX = {0}, gridY = {0};
        cuDeviceGetAttribute(gridX, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device);
        cuDeviceGetAttribute(gridY, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, device);
        this.maxGridX = gridX[0];
        this.maxGridY = gridY[0];

        int[] threadsPB = {0};
        cuDeviceGetAttribute(threadsPB, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
        this.maxThreadsPerBlock = threadsPB[0];

        log.info("GPU korlátok: maxGridX={}, maxGridY={}, maxThreadsPerBlock={}", maxGridX, maxGridY, maxThreadsPerBlock);
    }

    /**
     * Optimalizálja a grid és block méreteket N lebegőpontos elemhez.
     * @param N A feldolgozandó float elemek száma
     * @return Tömb: [gridX, gridY, blockSize]
     * @throws IllegalArgumentException ha a számított gridY meghaladja a GPU korlátot
     */
    public int[] optimize(int N) {
        int elements4 = (N + 3) / 4;
        int blockSize = Math.min(256, maxThreadsPerBlock);
        blockSize = (blockSize / 32) * 32;
        long totalBlocks = (elements4 + blockSize - 1L) / blockSize;
        int gridX = (int) Math.min(totalBlocks, maxGridX);
        int gridY = (int) ((totalBlocks + maxGridX - 1L) / maxGridX);
        if (gridY > maxGridY) {
            log.error("Igényelt gridY ({}) meghaladja a megengedettet ({})", gridY, maxGridY);
            throw new IllegalArgumentException("Igényelt gridY (" + gridY + ") túl nagy a GPU számára (max=" + maxGridY + ")");
        }
        log.debug("Optimalizálás N={} esetén: gridX={}, gridY={}, blockSize={}", N, gridX, gridY, blockSize);
        return new int[]{gridX, gridY, blockSize};
    }

    /** @return A GPU által támogatott maximális gridX dimenzió */
    public int getMaxGridX() { return maxGridX; }
    /** @return A GPU által támogatott maximális gridY dimenzió */
    public int getMaxGridY() { return maxGridY; }
    /** @return A GPU által támogatott maximális szálak száma blokkban */
    public int getMaxThreadsPerBlock() { return maxThreadsPerBlock; }
}