package org.example;

import org.apache.commons.math3.complex.Complex;

import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;
import java.io.File;
import java.io.IOException;

public class Main {
    public static void main(String[] args) {
        File audioFile = new File("C:/Users/Jason/Desktop/UNI/HC/Geheimnisvolle_Wellenlaengen.wav");
        int block_size = 16;
        try {
            AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(audioFile);
            int sampleRate = (int) audioInputStream.getFormat().getSampleRate();
            int numChannels = audioInputStream.getFormat().getChannels();
            long numFrames = audioInputStream.getFrameLength();
            int numBlocks = (int) (numFrames - block_size + 1);

            byte[] byteData = new byte[block_size];
            double[] data = new double[block_size];
            Complex[] audioFFT = new Complex[block_size];
            double[] frequencies = new double[block_size / 2];
            double[] magnitude = new double[block_size / 2];
            double[] allFrequencies = new double[numBlocks * block_size / 2];
            double[] allMagnitudes = new double[numBlocks * block_size / 2];
            double[] mainFreqs = new double[numBlocks * 2];

            long totalMemoryBytes = 0;

            totalMemoryBytes += byteData.length;
            totalMemoryBytes += data.length * Double.BYTES;
            totalMemoryBytes += audioFFT.length * (2 * Double.BYTES); // Each Complex object consists of two doubles
            totalMemoryBytes += frequencies.length * Double.BYTES;
            totalMemoryBytes += magnitude.length * Double.BYTES;
            totalMemoryBytes += allFrequencies.length * Double.BYTES;
            totalMemoryBytes += allMagnitudes.length * Double.BYTES;
            totalMemoryBytes += mainFreqs.length * Double.BYTES;

            double totalMemoryMb = totalMemoryBytes / (1024.0 * 1024.0);

            System.out.println("Total memory usage: " + totalMemoryMb + " MB");
        } catch (UnsupportedAudioFileException | IOException e) {
            e.printStackTrace();
        }
    }
}
