package laj.generators;

import laj.generators.utils.GeneratorParams;
import java.util.Random;

/**
 * Egyenletes eloszlású véletlenszám-generátor, amely a java.util.Random osztályt használja.
 * Ez az osztály egy paraméterként megadott tartományon (min, range) belül generál véletlen számokat,
 * amit a szülőosztály (CpuRandomGenerator) által definiált vector tömbbe tölt be.
 */
public class UniformJavaRandomGenerator extends CpuRandomGenerator {
    // Véletlenszám-generátor példány (java.util.Random)
    private final Random random;
    // Az előállított számok alsó határa
    private final int min;

    /**
     * Konstruktor.
     * 
     * @param params Paraméterek (minimum, mag/seed, stb.), amelyeket kívülről adunk át.
     *               A seed biztosítja, hogy újra előállíthatók legyenek ugyanazok a véletlenek.
     */
    public UniformJavaRandomGenerator(GeneratorParams params) {
        super(params);
        this.min = params.min(); // Minimum érték beállítása a generáláshoz
        this.random = new Random(params.seed()); // Véletlenszám-generátor inicializálása megadott maggal
    }

    /**
     * Véletlen egész számokat generál az előírt tartományban, 
     * és feltölti vele a vector tömböt.
     *
     * @param n Generálandó számok mennyisége. Feltételezi, hogy 'vector' mérete legalább n.
     */
    @Override
    public void generate(long n) {
        // Ciklus n hosszúságban, minden elemhez kisorsolunk egy számot
        for (int i = 0; i < n; i++) {
            // Egyenletesen eloszló számokat generál [min; min+range) tartományban
            vector[i] = random.nextInt(range) + min;
        }
    }
}