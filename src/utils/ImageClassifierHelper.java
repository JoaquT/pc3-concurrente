package utils;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class ImageClassifierHelper {

    public static final int COLOR_BINS = 512;
    public static final int CONV_FEATS = 54;
    public static final int TOTAL_FEATURES = COLOR_BINS + CONV_FEATS;

    // Filtros convolucionales 3x3
    public static final double[][][] KERNELS = {
        // 0: Horizontal Sobel (Detección de bordes horizontales)
        { {-1, -2, -1}, { 0,  0,  0}, { 1,  2,  1} },
        // 1: Vertical Sobel (Detección de bordes verticales)
        { {-1,  0,  1}, {-2,  0,  2}, {-1,  0,  1} },
        // 2: Laplaciano / Ridge (Detección de bordes y contornos isotrópicos)
        { { 0,  1,  0}, { 1, -4,  1}, { 0,  1,  0} },
        // 3: Diagonal 45° (Orientaciones diagonales)
        { {-2, -1,  0}, {-1,  0,  1}, { 0,  1,  2} },
        // 4: Sharpening (Altas frecuencias y textura de pelaje/alas)
        { { 0, -1,  0}, {-1,  5, -1}, { 0, -1,  0} },
        // 5: Emboss (Relieve y textura espacial tridimensional)
        { {-2, -1,  0}, {-1,  1,  1}, { 0,  1,  2} }
    };

    public static double[] extractFullFeatures(BufferedImage img) {
        int w = img.getWidth();
        int h = img.getHeight();

        // 1. Histograma de color de 512 bins (4x4x4 en RGB reducido a 8 niveles por canal)
        int[] bins = new int[COLOR_BINS];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = img.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                int bin = ((r / 32) * 64) + ((g / 32) * 8) + (b / 32);
                bins[bin]++;
            }
        }
        double totalPx = w * h;
        double[] colorHist = new double[COLOR_BINS];
        for (int i = 0; i < COLOR_BINS; i++) {
            colorHist[i] = bins[i] / totalPx;
        }

        // 2. Redimensionar a 32x32 para la etapa convolucional
        BufferedImage small = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2 = small.createGraphics();
        g2.drawImage(img, 0, 0, 32, 32, null);
        g2.dispose();

        // 3. Matriz de luminancia en escala de grises normalizada [0.0, 1.0]
        double[][] gray = new double[32][32];
        for (int y = 0; y < 32; y++) {
            for (int x = 0; x < 32; x++) {
                int rgb = small.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int gVal = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                gray[y][x] = (0.299 * r + 0.587 * gVal + 0.114 * b) / 255.0;
            }
        }

        // 4. Convoluciones
        double[] convFeat = computeConvolutions(gray);

        // 5. Concatenar vector final
        double[] combined = new double[TOTAL_FEATURES];
        System.arraycopy(colorHist, 0, combined, 0, COLOR_BINS);
        System.arraycopy(convFeat, 0, combined, COLOR_BINS, CONV_FEATS);
        return combined;
    }

    public static double[] computeConvolutions(double[][] gray) {
        double[] convFeat = new double[CONV_FEATS];
        double[][] pooledKernels = new double[KERNELS.length][9];

        // Procesamiento paralelo y concurrente de los filtros convolucionales
        java.util.stream.IntStream.range(0, KERNELS.length).parallel().forEach(k -> {
            double[][] kMat = KERNELS[k];
            double[][] map = new double[30][30];

            for (int y = 1; y < 31; y++) {
                for (int x = 1; x < 31; x++) {
                    double s = 0.0;
                    for (int ky = -1; ky <= 1; ky++) {
                        for (int kx = -1; kx <= 1; kx++) {
                            s += gray[y + ky][x + kx] * kMat[ky + 1][kx + 1];
                        }
                    }
                    map[y - 1][x - 1] = Math.max(0.0, s);
                }
            }

            int pIdx = 0;
            for (int by = 0; by < 3; by++) {
                for (int bx = 0; bx < 3; bx++) {
                    double m = 0.0;
                    for (int py = by * 10; py < (by + 1) * 10; py++) {
                        for (int px = bx * 10; px < (bx + 1) * 10; px++) {
                            if (map[py][px] > m) m = map[py][px];
                        }
                    }
                    pooledKernels[k][pIdx++] = m;
                }
            }
        });

        int idx = 0;
        double convSum = 0.0;
        for (int k = 0; k < KERNELS.length; k++) {
            for (int p = 0; p < 9; p++) {
                double val = pooledKernels[k][p];
                convFeat[idx++] = val;
                convSum += val;
            }
        }

        if (convSum > 0) {
            for (int i = 0; i < convFeat.length; i++) {
                convFeat[i] /= convSum;
            }
        }
        return convFeat;
    }

    public static RawImageData extractRawData(BufferedImage img) {
        int w = img.getWidth();
        int h = img.getHeight();

        int[] bins = new int[COLOR_BINS];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = img.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                int bin = ((r / 32) * 64) + ((g / 32) * 8) + (b / 32);
                bins[bin]++;
            }
        }
        double totalPx = w * h;
        double[] colorHist = new double[COLOR_BINS];
        for (int i = 0; i < COLOR_BINS; i++) colorHist[i] = bins[i] / totalPx;

        BufferedImage small = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2 = small.createGraphics();
        g2.drawImage(img, 0, 0, 32, 32, null);
        g2.dispose();

        double[] flatGray = new double[1024];
        for (int y = 0; y < 32; y++) {
            for (int x = 0; x < 32; x++) {
                int rgb = small.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int gVal = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                flatGray[y * 32 + x] = (0.299 * r + 0.587 * gVal + 0.114 * b) / 255.0;
            }
        }
        return new RawImageData(flatGray, colorHist);
    }

    public static class RawImageData {
        public final double[] flatGray; // 1024 valores
        public final double[] colorHist; // 512 valores
        public RawImageData(double[] flatGray, double[] colorHist) {
            this.flatGray = flatGray;
            this.colorHist = colorHist;
        }
    }

    public static double[] imageFileToVector(File file) throws Exception {
        BufferedImage img = ImageIO.read(file);
        if (img == null) throw new IOException("Formato corrupto o no soportado: " + file.getName());
        return extractFullFeatures(img);
    }

    public static double[] imageBytesToVector(byte[] bytes) throws Exception {
        ByteArrayInputStream bais = new ByteArrayInputStream(bytes);
        BufferedImage img = ImageIO.read(bais);
        if (img == null) throw new IOException("No se pudo decodificar la imagen enviada.");
        return extractFullFeatures(img);
    }
}