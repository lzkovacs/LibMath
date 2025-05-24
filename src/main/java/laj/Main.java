/*
 * Copyright (c) 2025. Hunc codicem scripsit Lajos, qui dicitur Kovács, ad suum solatium et eruditionem.
 */
package laj;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.JCudaDriver;
import laj.generators.DrawJavaRandomGenerator;
import laj.generators.SfmtJavaRandomGenerator;
import laj.generators.SortedCudaRandomGenerator;
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

        // Generate and log 20 random numbers using GPU
        generateAndLogRandomNumbers(context);

        // Generate and log 10 random numbers using SFMT
        generateAndLogSfmtRandomNumbers();

        // Generate and log 20 random numbers using DrawJavaRandomGenerator
        generateAndLogDrawRandomNumbers();

        // Destroy the context
        cuCtxDestroy(context);
    }

    /**
     * Generates 20 random numbers using SortedCudaRandomGenerator and logs them.
     * This method uses the SortedCudaRandomGenerator to generate random numbers in GPU memory,
     * then copies them back to CPU memory and logs them using Lombok logger.
     * 
     * @param context The CUDA context to use
     */
    @SuppressWarnings({"ExtractMethodRecommender", "LoggingSimilarMessage"})
    public static void generateAndLogRandomNumbers(CUcontext context) {
        // Create parameters for the generator
        GeneratorParams params = new GeneratorParams(
                Algorithm.SORTED_XORWOW,  // Sorted GPU algorithm
                42L,                      // Seed value
                20,                       // Vector size (20 numbers)
                true,                     // Transform enabled
                1,                        // Min value
                90,                       // Max value
                5                         // Block size
        );

        // Create the generator
        SortedCudaRandomGenerator generator = new SortedCudaRandomGenerator(params, context);

        try {
            // Generate 20 random numbers
            generator.generate(20);

            // Copy the numbers from GPU to CPU
            float[] hostArray = new float[20];
            Pointer hostPointer = Pointer.to(hostArray);
            cuMemcpyDtoH(hostPointer, generator.getDevPtr(), 20 * Sizeof.FLOAT);

            // Log the numbers
            log.debug("SortedCudaRandomGenerator: 20 darab random :");
            for (int i = 0; i < hostArray.length; i++) {
                log.debug("Number {}: {}", i + 1, hostArray[i]);
            }
        } finally {
            // Close the generator to release resources
            generator.close();
        }
    }

    /**
     * Generates 10 random numbers using SfmtJavaRandomGenerator and logs them.
     * This method uses the SfmtJavaRandomGenerator to generate random numbers in CPU memory
     * and logs them using Lombok logger.
     */
    public static void generateAndLogSfmtRandomNumbers() {
        // Create parameters for the generator
        GeneratorParams params = new GeneratorParams(
                Algorithm.SFMT,      // SFMT algorithm
                42L,                 // Seed value
                10,                  // Vector size (10 numbers)
                true,                // Transform enabled
                1,                   // Min value
                90,                  // Max value
                5                    // Block size
        );

        // Create the generator
        SfmtJavaRandomGenerator generator = new SfmtJavaRandomGenerator(params);

        // Generate 10 random numbers
        generator.generate(10);

        // Get the generated numbers
        float[] numbers = generator.getVector();

        // Log the numbers
        log.debug("SFMT 10 darab random:");
        for (int i = 0; i < numbers.length; i++) {
            log.debug("Number {}: {}", i + 1, numbers[i]);
        }
    }

    /**
     * Generates 20 random numbers using DrawJavaRandomGenerator and logs them.
     * This method uses the DrawJavaRandomGenerator to generate random numbers in CPU memory
     * and logs them using Lombok logger.
     */
    public static void generateAndLogDrawRandomNumbers() {
        // Create parameters for the generator
        GeneratorParams params = new GeneratorParams(
                Algorithm.DRAW_JAVA_RANDOM,  // DRAW_JAVA_RANDOM algorithm
                42L,                         // Seed value
                20,                          // Vector size (20 numbers)
                true,                        // Transform enabled
                1,                           // Min value
                90,                          // Max value
                5                            // Block size
        );

        // Create the generator
        DrawJavaRandomGenerator generator = new DrawJavaRandomGenerator(params);

        // Generate 20 random numbers
        generator.generate(20);

        // Get the generated numbers
        float[] numbers = generator.getVector();

        // Log the numbers
        log.debug("DrawJavaRandomGenerator: 20 darab random:");
        for (int i = 0; i < numbers.length; i++) {
            log.debug("Number {}: {}", i + 1, numbers[i]);
        }
    }
}
