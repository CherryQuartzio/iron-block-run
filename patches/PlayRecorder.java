package com.minerl.multiagent.recorder;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.microsoft.Malmo.Schemas.UnnamedGridDefinition;
import com.microsoft.Malmo.Utils.JSONWorldDataHelper;
import com.mojang.blaze3d.systems.RenderSystem;
import com.minerl.multiagent.RandomHelper;
import com.minerl.multiagent.env.EnvServer;
import net.minecraft.client.KeyboardListener;
import net.minecraft.client.Minecraft;
import net.minecraft.client.MouseHelper;
import net.minecraft.client.ReplaySender;
import net.minecraft.client.entity.player.ClientPlayerEntity;
import net.minecraft.client.gui.screen.inventory.HorseInventoryScreen;
import net.minecraft.client.gui.screen.inventory.InventoryScreen;
import net.minecraft.client.shader.Framebuffer;
import net.minecraft.server.integrated.IntegratedServer;
import nu.pattern.OpenCV;
import org.apache.commons.io.FilenameUtils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoWriter;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

import static com.microsoft.Malmo.Utils.CraftingHelper.dumpMinecraftObjectRules;
import static org.lwjgl.opengl.GL11.*;

public class PlayRecorder {
    private static PlayRecorder instance = null;
    private String prefix;
    private String filename;
    private String azurePath;
    private FileWriter actionsWriter;
    private int tickCounter = -1;
    private VideoWriter videoWriter;
    private Framebuffer fbo;
    private int width;
    private int height;
    private int fps = 20;
    private boolean recording = false;
    private boolean pausedLastTick = false;
    private String userid;
    private String version;
    private Mat lastFrame;
    private byte[] lastImageBytes;
    private ByteBuffer imageByteBuffer;
    // each video will be no longer than this (currently set to five minutes)
    private static final int maxDuration = 6000;
    // disable autosaves - the world is would be saved anyways every maxDuration steps
    private static final int saveStatePeriod = -1;
    private String episodeid = RandomHelper.getRandomHexString();
    Minecraft mc = Minecraft.getInstance();
    JsonElement mouseState;
    JsonElement keyboardState;


    public static PlayRecorder getInstance() {
        if (instance == null) {
            instance = new PlayRecorder(Paths.get(System.getProperty("user.dir"), "recordings").toString());
        }
        return instance;
    }

    public byte[] getLastImageBytes() {
        return lastImageBytes;
    }

    public boolean isRecording() {
        return recording;
    }

    public PlayRecorder(String prefix) {
        this.prefix = prefix;
    }

    public void start() {
        tickCounter = 0;

        try {
            // These hoops are necessary because openpnp does not include ffmpeg bindings on linux due
            // to licensing concerns. So, on linux, an external library must be installed (apt install libopencv3.2-jni)
            // and loaded
            width = mc.getFramebuffer().framebufferWidth;
            height = mc.getFramebuffer().framebufferHeight;
            if (!mc.gameSettings.disableRecorder) {
                width = 640;
                height = 360;
            }
            // this.fbo = new Framebuffer(width, height, true, Minecraft.IS_RUNNING_ON_MAC);
            imageByteBuffer = ByteBuffer.allocateDirect(width * height * 3);
            lastImageBytes = new byte[imageByteBuffer.capacity()];
            resolveImageRequests();
            if (!mc.gameSettings.disableRecorder) {
                if (System.getProperty("os.name").equals("Linux")) {
                    System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
                } else {
                    OpenCV.loadLocally();
                }
                Properties versionProp = new Properties();
                versionProp.load(this.getClass().getClassLoader().getResourceAsStream("version.properties"));
                version = versionProp.getProperty("version");
                userid = System.getenv().getOrDefault("MINEREC_UID", "unnamed");

                if (lastFrame == null) {
                    lastFrame = new Mat(height, width, CvType.CV_8UC3);
                }
                String playerName = mc.player.getName().getString();
                filename = Paths.get(this.prefix, version, playerName + "-" + episodeid + "-" + DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss").format(LocalDateTime.now())).toString();
                azurePath = "";
                Files.createDirectories(Paths.get(FilenameUtils.getFullPath(filename)));
                videoWriter = new VideoWriter(filename + ".mp4", VideoWriter.fourcc('x', '2', '6', '4'),
                        fps, new Size(width, height), true);
                if (!videoWriter.isOpened()) {
                    System.out.println("Cannot open VideoWriter! Here's opencv build info");
                    System.out.println(Core.getBuildInformation());
                    videoWriter.release();
                    throw new IllegalArgumentException("VideoWriter Exception: VideoWriter not opened,"
                            + "check parameters.");
                }
                actionsWriter = new FileWriter(filename + ".jsonl", true);
                FileWriter optionsWriter = new FileWriter(filename + "-options.json", true);
                optionsWriter.write(getOptionsJson().toString());
                optionsWriter.close();

                IntegratedServer is = mc.getIntegratedServer();
                if (is != null) {
                    is.saveAndUploadWorld(filename + ".zip");
                    is.setUploadPath(azurePath);
                    is.setAutosavePeriod(saveStatePeriod);
                    is.setWorldZipPrefix(filename);
                }

            }
            if (ReplaySender.getInstance().getMode() == ReplaySender.Mode.OFF) {
                ReplaySender.getInstance().sendFromEnv();
            }
            System.out.println("Starting new video " + filename);
            recording = true;
            pausedLastTick = false;
            // clearMouseKeyboardState();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private JsonObject getOptionsJson() {
        Minecraft mc = Minecraft.getInstance();
        JsonObject retVal = mc.gameSettings.optionsJson();
        retVal.addProperty("framebufferToWindowRatio", mc.getMainWindow().getFramebufferWidth() / mc.getMainWindow().getWidth());
        retVal.addProperty("windowWidth", mc.getMainWindow().getWidth());
        retVal.addProperty("windowHeight", mc.getMainWindow().getHeight());
        retVal.addProperty("framebufferWidth", mc.getMainWindow().getFramebufferWidth());
        retVal.addProperty("framebufferHeight", mc.getMainWindow().getFramebufferHeight());
        return retVal;
    }

    public void tick() {
        Minecraft mc = Minecraft.getInstance();
        ClientPlayerEntity player = mc.player;
        if (player == null || !player.isAlive()) {
            if (recording) {
                finishAndResetEpisode();
            }
            return;
        }
        if (!recording) {
            start();
        }
        if (pausedLastTick || mc.isGamePaused()) {
            // Pause recording for one tick after the game
            // is unpaused to prevent capture of menu frames
            // When paused, clear mouse and keyboard state
            mc.mouseHelper.setHumanInput(true);
            clearMouseKeyboardState();
        } else {
            mc.mouseHelper.setHumanInput(false);
            recordTickImpl();
        }
        pausedLastTick = mc.isGamePaused();
        if (tickCounter == maxDuration) {
            finish();
        }
    }

    private void clearMouseKeyboardState() {
        Minecraft mc = Minecraft.getInstance();
        mc.mouseHelper.clearState();
        mc.keyboardListener.clearState();
    }

    private void recordTickImpl() {
        Gson gson = new Gson();
        int capacity = width * height * 3;
        resolveImageRequests();
        if (!mc.gameSettings.disableRecorder) {
            ByteBuffer imgBuffer = ByteBuffer.allocateDirect(capacity);
            getRGBFrame(imgBuffer);
            byte[] imgBytes = new byte[capacity];
            imgBuffer.get(imgBytes);
            lastFrame.put(0, 0, imgBytes);
            Core.flip(lastFrame, lastFrame, 0);
            Imgproc.cvtColor(lastFrame, lastFrame, Imgproc.COLOR_BGR2RGB);
            videoWriter.write(lastFrame);
            if (mc.gameSettings.envPort == 0) {
                mouseState = gson.toJsonTree(mc.mouseHelper.getState());
                keyboardState = gson.toJsonTree(mc.keyboardListener.getState());
            }
            JsonObject actions = new JsonObject();
            actions.add("mouse", mouseState);
            actions.add("keyboard", keyboardState);
            actions.addProperty("isGuiOpen", mc.currentScreen != null);
            actions.addProperty("isGuiInventory",
                    mc.currentScreen != null && (
                            mc.currentScreen instanceof InventoryScreen ||
                            mc.currentScreen instanceof HorseInventoryScreen
                    ));
            actions.addProperty("hotbar", mc.player.inventory.currentItem );
            actions.addProperty("yaw", mc.player.rotationYaw);
            actions.addProperty("pitch", mc.player.rotationPitch);
            actions.addProperty("xpos", mc.player.getPosX());
            actions.addProperty("ypos", mc.player.getPosY());
            actions.addProperty("zpos", mc.player.getPosZ());
            actions.addProperty("tick", tickCounter);
            actions.addProperty("milli", System.currentTimeMillis());
            actions.add("inventory", EnvServer.getInventoryJson());
            if (mc.getIntegratedServer() != null) {
                actions.addProperty("serverTick", mc.getIntegratedServer().getTickCounter());
                actions.addProperty("serverTickDurationMs", mc.getIntegratedServer().getTickTimeRaw());
            }
            if (ReplaySender.getInstance().getMode() == ReplaySender.Mode.EXEC_CMD && mc.gameSettings.envPort == 0) {
                ReplaySender.getInstance().addAction(actions);
            }
            // System.out.println("tick " + tickCounter + ", " + getStats());
            actions.add("stats", getStats());

            try {
                actionsWriter.write(actions.toString());
                actionsWriter.write("\n");
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        mouseState = null;
        keyboardState = null;
        tickCounter++;
        synchronized (this) {
            this.notifyAll();
        }
    }

    private JsonObject getStats() {
        ClientPlayerEntity player = Minecraft.getInstance().player;
        JsonObject infoJson = new JsonObject();
        if (player != null) {
            JSONWorldDataHelper.buildAllStats(infoJson, player);
        }
        return infoJson;
    }


    private void upload() {
        AzureUpload.uploadAsync(filename + ".jsonl", azurePath);
        AzureUpload.uploadAsync(filename + "-options.json", azurePath);
        AzureUpload.uploadAsync(filename + ".mp4", azurePath);
    }

    public void finish() {
        tickCounter = -1;
        if (!recording) {
            return;
        }
        try {
            recording = false;
            mc.mouseHelper.setHumanInput(true);
            // lastImageBytes = null;
            if (!Minecraft.getInstance().gameSettings.disableRecorder) {
                System.out.println("Finalizing the video");
                actionsWriter.close();
                videoWriter.write(lastFrame);
                videoWriter.release();
                upload();
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void finishAndResetEpisode() {
        finish();
        lastFrame = null;
        episodeid = RandomHelper.getRandomHexString();
    }

    /** Soft persistent reset: keep recording and integrated server, reset tick counter only. */
    public void softResetEpisode() {
        tickCounter = 0;
        episodeid = RandomHelper.getRandomHexString();
        pausedLastTick = false;
        recording = true;
        if (ReplaySender.getInstance().getMode() == ReplaySender.Mode.OFF) {
            ReplaySender.getInstance().sendFromEnv();
        }
        synchronized (this) {
            this.notifyAll();
        }
        System.out.println("[Persistent] PlayRecorder soft episode reset");
    }

    private void getRGBFrame(ByteBuffer buffer) {
        RenderSystem.pushMatrix();
        mc.getFramebuffer().framebufferRender(width, height);
        RenderSystem.popMatrix();
        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, buffer);
    }

    public int getTickCounter() {
        return tickCounter;
    }

    public void setMouseKeyboardState(MouseHelper.State mouseState, KeyboardListener.State keyboardState) {
        Gson gson = new Gson();
        this.mouseState = gson.toJsonTree(mouseState);
        this.keyboardState = gson.toJsonTree(keyboardState);
    }

    private void resolveImageRequests() {
        mc.getProfiler().startSection("resolveImageRequests");
        mc.getProfiler().endStartSection("getRBGFrame");
        imageByteBuffer.rewind();
        getRGBFrame(imageByteBuffer);
        mc.getProfiler().endStartSection("getBytes");
        imageByteBuffer.get(lastImageBytes);
        mc.getProfiler().endSection();
        mc.getProfiler().endSection();
    }
}
