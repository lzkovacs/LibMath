/*
 * Copyright (c) 2024-2025. Hunc codicem scripsit Lajos, qui dicitur Kovács, ad suum solatium et eruditionem.
 */
package generators.generator_utils;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
/**
 * Tesztosztály a Seeder osztályhoz.
 * Ez az osztály a Seeder osztály alapvető funkcionalitását teszteli, beleértve:
 * - Konstruktor különböző SeedSource értékekkel
 * - generateIntSeed() metódus
 * - generateLongSeed() metódus
 * - Hibakezelés null forrás esetén
 */
class SeederTest {
    private Seeder seederWithNanoTime;
    @BeforeEach
    void setUp() {
        // Létrehoz egy Seedert NANOTIME forrással az alap tesztekhez
        seederWithNanoTime = new Seeder(SeedSource.NANOTIME);
    }
    @Test
    void constructor_WithValidSource_ShouldCreateInstance() {
        // Tesztelés minden érvényes SeedSource értékkel
        for (SeedSource source : SeedSource.values()) {
            Seeder seeder = new Seeder(source);
            assertNotNull(seeder);
            assertEquals(source, seeder.getSource());
        }
    }
    @Test
    void constructor_WithNullSource_ShouldThrowNullPointerException() {
        // Tesztelés null forrással
        Exception exception = assertThrows(NullPointerException.class, () -> new Seeder(null));
        assertTrue(exception.getMessage().contains("null"));
    }
    @Test
    void generateIntSeed_WithNanoTimeSource_ShouldReturnIntValue() {
        // Teszt NANOTIME forrással
        int seed = seederWithNanoTime.generateIntSeed();
        // Nem tudjuk előre az értéket, de ellenőrizzük, hogy nem 0
        assertNotEquals(0, seed);
    }
    @Test
    void generateLongSeed_WithNanoTimeSource_ShouldReturnLongValue() {
        // Teszt NANOTIME forrással
        long seed = seederWithNanoTime.generateLongSeed();
        // Nem tudjuk előre az értéket, de ellenőrizzük, hogy nem 0
        assertNotEquals(0L, seed);
    }
    @Test
    void generateIntSeed_WithExternalSource_ShouldReturnValue() {
        // Tesztek külső forrásokkal
        // Megjegyzés: Ez a teszt valójában nem kapcsolódik külső szolgáltatáshoz,
        // csak ellenőrzi, hogy a metódus nem dob kivételt
        for (SeedSource source : SeedSource.values()) {
            if (source != SeedSource.NANOTIME) {
                Seeder seeder = new Seeder(source);
                int seed = seeder.generateIntSeed();
                // Nem tudjuk előre az értéket, de ellenőrizzük, hogy nem 0
                // Megjegyzés: Ha csatlakozási hiba van, visszaesik nanoTime-ra
                assertNotEquals(0, seed);
            }
        }
    }
    @Test
    void generateLongSeed_WithExternalSource_ShouldReturnValue() {
        // Tesztek külső forrásokkal
        // Megjegyzés: Ez a teszt valójában nem kapcsolódik külső szolgáltatáshoz,
        // csak ellenőrzi, hogy a metódus nem dob kivételt
        for (SeedSource source : SeedSource.values()) {
            if (source != SeedSource.NANOTIME) {
                Seeder seeder = new Seeder(source);
                long seed = seeder.generateLongSeed();
                // Nem tudjuk előre az értéket, de ellenőrizzük, hogy nem 0
                // Megjegyzés: Ha csatlakozási hiba van, visszaesik nanoTime-ra
                assertNotEquals(0L, seed);
            }
        }
    }
}