package laj.generators.utils;


import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.Arrays;
import java.util.Properties;

/**
 * Seeder osztály, amely különböző forrásokból képes véletlen seed értékeket generálni.
 * Támogatott források: SYSTEM, RANDOM_ORG, ANU_QUANTUM, FOURMILAB
 */
@Log4j2
public class Seeder {
    @Getter
    private final SeedSource source;
    private static final String PROPERTIES_FILE = "seeder.properties";
    private static final String RANDOM_ORG_URL_KEY = "random.org.url";
    private static final String ANU_QUANTUM_URL_KEY = "anu.quantum.url";
    private static final String FOURMILAB_URL_KEY = "fourmilab.url";
    @Getter
    private String randomOrgUrl;
    @Getter
    private String anuQuantumUrl;
    @Getter
    private String fourmiLabUrl;

    /**
     * Konstruktor, amely beállítja a seed forrást és betölti a konfigurációs fájlt.
     * @param source A seed forrása (SYSTEM, RANDOM_ORG, ANU_QUANTUM, FOURMILAB)
     */
    public Seeder(SeedSource source) {
        if (source == null) {
            throw new NullPointerException("A forrás nem lehet null");
        }
        this.source = source;
        try {
            Properties props = loadProperties();
            randomOrgUrl = props.getProperty(RANDOM_ORG_URL_KEY);
            anuQuantumUrl = props.getProperty(ANU_QUANTUM_URL_KEY);
            fourmiLabUrl = props.getProperty(FOURMILAB_URL_KEY);
        } catch (IOException e) {
            log.error("Hiba a seeder.properties betöltésekor", e);
            // Alapértelmezett URL-ek beállítása
            randomOrgUrl = "https://www.random.org/integers/";
            anuQuantumUrl = "https://qrng.anu.edu.au/API/jsonI.php";
            fourmiLabUrl = "https://www.fourmilab.ch/cgi-bin/Hotbits";
        }
    }

    /**
     * Generál egy véletlen egész számot a beállított forrásból.
     * @return A generált véletlen egész szám
     */
    public int generateIntSeed() {
        byte[] bytes = getRandomBytes(4);
        int result = 0;
        for (int i = 0; i < 4; i++) {
            result = (result << 8) | (bytes[i] & 0xFF);
        }
        log.info("Generált int seed: {} forrás: {}", result, source);
        return result;
    }

    /**
     * Generál egy véletlen long számot a beállított forrásból.
     * @return A generált véletlen long szám
     */
    public long generateLongSeed() {
        byte[] bytes = getRandomBytes(8);
        long result = 0;
        for (int i = 0; i < 8; i++) {
            result = (result << 8) | (bytes[i] & 0xFF);
        }
        log.info("Generált long seed: {} forrás: {}", result, source);
        return result;
    }

    /**
     * Lekér megadott számú véletlen bájtot a beállított forrásból.
     * @param numBytes A kért bájtok száma
     * @return A véletlen bájtok tömbje
     */
    private byte[] getRandomBytes(int numBytes) {
        return switch (source) {
            case SYSTEM -> {
                byte[] bytes = new byte[numBytes];
                new java.util.Random().nextBytes(bytes);
                yield bytes;
            }
            case NANOTIME -> {
                byte[] bytes = new byte[numBytes];
                long nanoTime = System.nanoTime();
                for (int i = 0; i < numBytes; i++) {
                    bytes[i] = (byte) ((nanoTime >> (i % 8 * 8)) & 0xFF);
                }
                yield bytes;
            }
            case RANDOM_ORG -> getRandomOrgBytes(numBytes);
            case ANU_QUANTUM -> getANUQuantumBytes(numBytes);
            case FOURMILAB -> getFourmilabHotBits(numBytes);
        };
    }

    /**
     * Lekér véletlen bájtokat a Random.org szolgáltatástól.
     * @param numBytes A kért bájtok száma
     * @return A véletlen bájtok tömbje
     */
    private byte[] getRandomOrgBytes(int numBytes) {
        try {
            String url = String.format(randomOrgUrl, numBytes, 0, 255);
            HttpClient client = HttpClient.newHttpClient();
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .GET()
                    .build();
            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
            String[] numbers = response.body().split("\n");
            byte[] result = new byte[numBytes];
            for (int i = 0; i < numBytes && i < numbers.length; i++) {
                result[i] = (byte) Integer.parseInt(numbers[i].trim());
            }
            return result;
        } catch (Exception e) {
            log.error("Hiba a Random.org lekérdezésekor: {}", e.getMessage());
            // Fallback a rendszer véletlenszám-generátorára
            byte[] fallbackBytes = new byte[numBytes];
            new java.util.Random().nextBytes(fallbackBytes);
            return fallbackBytes;
        }
    }

    /**
     * Lekér véletlen bájtokat az ANU Quantum Random Number Generator szolgáltatástól.
     * @param numBytes A kért bájtok száma
     * @return A véletlen bájtok tömbje
     */
    private byte[] getANUQuantumBytes(int numBytes) {
        try {
            String url = String.format(anuQuantumUrl, numBytes);
            HttpClient client = HttpClient.newHttpClient();
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .GET()
                    .build();
            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
            // Egyszerű JSON feldolgozás a válaszból
            String jsonResponse = response.body();

            // Ellenőrizzük, hogy a válasz tartalmazza-e a várt JSON struktúrát
            if (!jsonResponse.contains("\"data\":[") || !jsonResponse.contains("]")) {
                log.error("Érvénytelen ANU quantum válasz: nincs megfelelő karakter ([ vagy ]) a válaszban: {}", jsonResponse);
                throw new IllegalStateException("Érvénytelen ANU quantum válasz: nincs megfelelő karakter ([ vagy ])");
            }

            String dataStr = jsonResponse.split("\"data\":\\[")[1].split("]")[0];
            String[] numbers = dataStr.split(",");
            byte[] result = new byte[numBytes];
            for (int i = 0; i < numBytes && i < numbers.length; i++) {
                result[i] = (byte) Integer.parseInt(numbers[i].trim());
            }
            return result;
        } catch (Exception e) {
            log.error("Hiba az ANU Quantum lekérdezésekor: {}", e.getMessage());
            // Fallback a rendszer véletlenszám-generátorára
            byte[] fallbackBytes = new byte[numBytes];
            new java.util.Random().nextBytes(fallbackBytes);
            return fallbackBytes;
        }
    }

    /**
     * Lekér véletlen bájtokat a Fourmilab HotBits szolgáltatástól.
     * @param numBytes A kért bájtok száma
     * @return A véletlen bájtok tömbje
     */
    private byte[] getFourmilabHotBits(int numBytes) {
        try {
            String url = String.format(fourmiLabUrl, numBytes);
            HttpClient client = HttpClient.newHttpClient();
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .GET()
                    .build();
            HttpResponse<byte[]> response = client.send(request, HttpResponse.BodyHandlers.ofByteArray());
            return Arrays.copyOf(response.body(), numBytes);
        } catch (Exception e) {
            log.error("Hiba a Fourmilab HotBits lekérdezésekor: {}", e.getMessage());
            // Fallback a rendszer véletlenszám-generátorára
            byte[] fallbackBytes = new byte[numBytes];
            new java.util.Random().nextBytes(fallbackBytes);
            return fallbackBytes;
        }
    }

    /**
     * Betölti a konfigurációs fájlt.
     * @return A betöltött Properties objektum
     * @throws IOException Ha hiba történik a fájl betöltésekor
     */
    private Properties loadProperties() throws IOException {
        Properties props = new Properties();
        try (InputStream input = getClass().getClassLoader().getResourceAsStream(PROPERTIES_FILE)) {
            if (input == null) {
                throw new IOException("Nem található a " + PROPERTIES_FILE + " fájl");
            }
            props.load(input);
        }
        return props;
    }
}
