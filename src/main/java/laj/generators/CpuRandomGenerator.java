package laj.generators;

import laj.generators.utils.GeneratorParams;
import lombok.Getter;

/**
 * Absztrakt osztály véletlenszám-generátorok CPU-n történő implementálásához.
 * Tartalmazza az alapvető mezőket és konstrukciót, valamint elvárt a véletlenszám-generáló metódus implementációja.
 *
 */
@Getter
public abstract class CpuRandomGenerator {
    /**
     * Egy tömb, amely a generált véletlen float értékeket tárolja.
     */
    protected final float[] vector;

    /**
     * A generált számok tartományának nagysága. 
     * (max - min + 1)
     */
    protected final int range;

    /**
     * Konstruktor, mely beállítja a vektorméretet és tartományt a paraméterek alapján.
     * 
     * @param params paraméterobjektum, amely a vektor méretét és a tartomány határait tartalmazza
     */
    protected CpuRandomGenerator(GeneratorParams params) {
        // Inicializálja a vektort a megadott mérettel
        vector = new float[params.vectorSize()];
        // Beállítja a generált értékek tartományát
        range = params.max() - params.min() + 1;
    }
    
    /**
     * Véletlenszámokat generál és feltölti a {@code vector} tömböt.
     * Az implementációt a konkrét leszármazott osztályok valósítják meg.
     * 
     * @param n a generálandó véletlenszámok száma
     */
    public abstract void generate(long n);
}