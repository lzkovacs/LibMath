/*
 * Copyright (c) 2025. Hunc codicem scripsit Lajos, qui dicitur Kovács, ad suum solatium et eruditionem.
 */
package laj;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.JCudaDriver;
import laj.generators.ScaledCudaRandomGenerator;
import laj.generators.utils.Algorithm;
import laj.generators.utils.GeneratorParams;
import lombok.extern.slf4j.Slf4j;

import static jcuda.driver.JCudaDriver.*;

@Slf4j
public class Main {
    public static void main(String[] args) {
        // Initialize JCuda
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);

        // Create a CUDA context
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        // Generate and log 10 random numbers
        generateAndLogRandomNumbers(context);

        // Destroy the context
        cuCtxDestroy(context);
    }

    /**
     * Generates 10 random numbers using ScaledCudaRandomGenerator and logs them.
     * This method uses the ScaledCudaRandomGenerator to generate random numbers in GPU memory,
     * then copies them back to CPU memory and logs them using Lombok logger.
     * 
     * @param context The CUDA context to use
     */
    public static void generateAndLogRandomNumbers(CUcontext context) {
        // Create parameters for the generator
        GeneratorParams params = new GeneratorParams(
                Algorithm.SCALE_XORWOW,  // Scaled GPU algorithm
                42L,                     // Seed value
                10,                      // Vector size (10 numbers)
                true,                    // Transform enabled
                1,                       // Min value
                90,                      // Max value
                5                        // Block size
        );

        // Create the generator
        ScaledCudaRandomGenerator generator = new ScaledCudaRandomGenerator(params, context);

        try {
            // Generate 10 random numbers
            generator.generate(10);

            // Copy the numbers from GPU to CPU
            float[] hostArray = new float[10];
            Pointer hostPointer = Pointer.to(hostArray);
            cuMemcpyDtoH(hostPointer, generator.getDevPtr(), 10 * Sizeof.FLOAT);

            // Log the numbers
            log.debug("Generated 10 random numbers:");
            for (int i = 0; i < hostArray.length; i++) {
                log.debug("Number {}: {}", i + 1, hostArray[i]);
            }
        } finally {
            // Close the generator to release resources
            generator.close();
        }
    }
}
