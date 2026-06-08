package net.minecraft.client;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.minerl.multiagent.RandomHelper;
import com.minerl.multiagent.recorder.ZipUtil;
import net.minecraft.client.gui.screen.DirtMessageScreen;
import net.minecraft.client.gui.screen.MainMenuScreen;
import net.minecraft.client.util.InputMappings;
import net.minecraft.util.text.TranslationTextComponent;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

public class ReplaySender {
    private static ReplaySender instance = new ReplaySender();
    private List<JsonObject> actions;
    private boolean firstTick = false;
    private int serverStartTick;
    private JsonObject lastAction = null;

    public enum Mode {
        REPLAY_FILE, OFF, EXEC_CMD
    }

    private Mode mode = Mode.OFF;

    public static ReplaySender getInstance() {
        return instance;
    }

    public synchronized void sendFromFile(String recordingFile, int serverStartTick) {
        // load game settings?
        // open and load jsonl file
        if (mode != Mode.OFF) {
            throw new RuntimeException("ReplaySender is already sending the replay, need to stop it first");
        }
        Gson gson = new Gson();
        try {
            this.serverStartTick = serverStartTick;
            actions = Files.readAllLines(new File(recordingFile).toPath(), Charset.defaultCharset())
                    .stream()
                    .map(s -> gson.fromJson(s, JsonObject.class))
                    .filter(j -> j.get("serverTick").getAsInt() >= (serverStartTick - 1))
                    .collect(Collectors.toList());


        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        Minecraft mc = Minecraft.getInstance();
        mc.mouseHelper.setHumanInput(false);
        mode = Mode.REPLAY_FILE;
        firstTick = true;
        // EnvServer.execActions("camera 0 0", 0);
    }

    public synchronized void sendFromEnv() {
        if (mode != Mode.OFF) {
            throw new RuntimeException("ReplaySender is already sending the replay, need to stop it first");
        }
        Minecraft mc = Minecraft.getInstance();
        mc.mouseHelper.setHumanInput(false);
        actions = new LinkedList<>();
        mode = Mode.EXEC_CMD;
        firstTick = true;
        // EnvServer.execActions("camera 0 0", 0);
    }

    public void tick() {
        if (mode == Mode.OFF) {
            return;
        }
        Minecraft mc = Minecraft.getInstance();
        // PlayRecorder.getInstance().reseedEntities();

        if (mode == Mode.REPLAY_FILE && actions.size() == 0) {
            stop();
        }
        if (!mc.mouseHelper.getHumanInput()) {
            mc.getProfiler().startSection("waitForAction");


            try {
                synchronized (this) {
                    while (mode == Mode.EXEC_CMD && actions.size() == 0) {
                        this.wait();
                    }
                }
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }

            mc.getProfiler().endSection();

            if (firstTick) {
                onFirstTick();
                firstTick = false;
            }
            mc.getProfiler().startSection("execAction");
            if (actions.size() > 0) {
                JsonObject action = actions.remove(0);
                execAction(action);
            }
            mc.getProfiler().endSection();
        } else {
            actions.clear();
        }
    }

    private void onFirstTick() {
    }

    public void addAction(JsonObject e) {
        actions.add(e);
        synchronized (this) {
            this.notifyAll();
        }
    }

    public void addAction(MouseHelper.State mouseState, KeyboardListener.State keyboardState) {
        Gson gson = new Gson();
        JsonObject jo = new JsonObject();
        jo.add("mouse", gson.toJsonTree(mouseState));
        jo.add("keyboard", gson.toJsonTree(keyboardState));
        jo.addProperty("isGuiOpen", false);
        addAction(jo);
    }

    public synchronized void clearEpisodeState() {
        if (mode == Mode.OFF) {
            return;
        }
        actions.clear();
        firstTick = true;
        Minecraft mc = Minecraft.getInstance();
        mc.mouseHelper.setHumanInput(false);
        synchronized (this) {
            this.notifyAll();
        }
    }

    public void stop() {
        if (mode != Mode.OFF) {
            System.out.println("*** Stopping the replay, returning control to the inputs");
            Minecraft mc = Minecraft.getInstance();
            mc.execute( () -> {
                        mc.mouseHelper.setHumanInput(true);
                        mc.world.sendQuittingDisconnectingPacket();
                        mc.unloadWorld(new DirtMessageScreen(new TranslationTextComponent("menu.savingLevel")));
                        mc.displayGuiScreen(new MainMenuScreen());
                    });
            mode = Mode.OFF;
            synchronized (this) {
                this.notifyAll();
            }
        }
    }

    private static KeyboardListener.State getKeyboardState(JsonObject action) {
        Gson gson = new Gson();
        return gson.fromJson(action.getAsJsonObject("keyboard"), KeyboardListener.State.class);
    }

    private static MouseHelper.State getMouseState(JsonObject action) {
        Gson gson = new Gson();
        return gson.fromJson(action.getAsJsonObject("mouse"), MouseHelper.State.class );
    }

    private void execAction(JsonObject action) {
        Minecraft mc = Minecraft.getInstance();
        MouseHelper.State mouseState = getMouseState(action);
        KeyboardListener.State keyboardState = getKeyboardState(action);
        // boolean newIsGuiOpen = action.get("isGuiOpen").getAsBoolean();
        int mods = getModifiers(keyboardState.keys);

        if (action.get("serverTick") == null || action.get("serverTick").getAsInt() >= serverStartTick) {
            if (firstTick) {
                onFirstTick();
                firstTick = false;
            }
            if (mode == Mode.REPLAY_FILE) {
                checkPlayerPosition();
                checkPlayerInventory();
            }
            execKeyboard(keyboardState);
            execMouseMove(mouseState);
            execMouseButtons(mouseState.buttons, mods);
            execMouseScroll(mouseState.dwheel);
        }
        lastAction = action;
    }

    private static void setPlayerPosition(JsonObject action) {
        if (action == null) {
            return;
        }
        Minecraft mc = Minecraft.getInstance();
        mc.player.rotationYaw = action.get("yaw").getAsFloat();
        mc.player.rotationPitch = action.get("pitch").getAsFloat();
        mc.player.setPosition(action.get("xpos").getAsDouble(), action.get("ypos").getAsDouble(), action.get("zpos").getAsDouble());
    }

    private void checkPlayerPosition() {
        JsonObject action = lastAction;
        if (action == null) {
            return;
        }
        double maxCoordDiff = 1;
        double maxAngleDiff = 1;
        Minecraft mc = Minecraft.getInstance();

        if (Math.abs(mc.player.getPosX() - lastAction.get("xpos").getAsDouble()) > maxCoordDiff ||
            Math.abs(mc.player.getPosY() - lastAction.get("ypos").getAsDouble()) > maxCoordDiff ||
            Math.abs(mc.player.getPosZ() - lastAction.get("zpos").getAsDouble()) > maxCoordDiff ||
            Math.abs(mc.player.rotationYaw - lastAction.get("yaw").getAsDouble()) > maxAngleDiff ||
            Math.abs(mc.player.rotationPitch - lastAction.get("pitch").getAsDouble()) > maxAngleDiff
        ) {
            // stop();
            System.out.println("position or angle have drifted, should stop the replay");
        }
    }

    private void checkPlayerInventory() {
        return;
    }

    private int getModifiers(Set<String> keys) {
        int retval = 0;
        if (keys.contains("key.keyboard.left.shift") || keys.contains("key.keyboard.right.shift")) {
            retval |= 1;
        }
        return retval;
    }

    private double clip(double v, double lower, double upper) {
        return Math.min(Math.max(v, lower), upper);
    }

    private void execMouseMove(MouseHelper.State mouseState) {
        Minecraft mc = Minecraft.getInstance();
        MouseHelper mh = mc.mouseHelper;
        MainWindow mw = mc.getMainWindow();
        if (mouseState.dx != 0 || mouseState.dy != 0) {
            double scaleFactor = mh.getScaleFactor();
            double newMouseX = mh.getMouseX() + mouseState.dx * scaleFactor;
            double newMouseY = mh.getMouseY() + mouseState.dy * scaleFactor;
            if (mc.currentScreen != null) {
                newMouseX = clip(newMouseX, 0, mw.getWidth());
                newMouseY = clip(newMouseY, 0, mw.getHeight());
            }
            mh.cursorPosCallbackImpl(newMouseX, newMouseY);
        }
    }

    private void execMouseButtons(Set<Integer> buttons, int mods) {
        Minecraft mc = Minecraft.getInstance();
        Set<Integer> pressedButtons = new HashSet<>();
        Set<Integer> releasedButtons = new HashSet<>();
        pressedButtons.addAll(buttons);
        if (lastAction != null) {
            pressedButtons.removeAll(getMouseState(lastAction).buttons);
            releasedButtons.addAll(getMouseState(lastAction).buttons);
            releasedButtons.removeAll(buttons);
        }
        for (int button : pressedButtons) {
            mc.mouseHelper.mouseButtonCallbackImpl(mc.getMainWindow().getHandle(), button, 1, 0);
        }
        for (int button : releasedButtons) {
            mc.mouseHelper.mouseButtonCallbackImpl(mc.getMainWindow().getHandle(), button, 0, 0);
        }
        mc.mouseHelper.getState();
    }

    private void execKeyboard(KeyboardListener.State state) {
        Minecraft mc = Minecraft.getInstance();
        Set<String> newlyHitKeys = new HashSet<>();
        Set<String> releasedKeys = new HashSet<>();
        int mods = 0;
        newlyHitKeys.addAll(state.keys);
        if (lastAction != null) {
            newlyHitKeys.removeAll(getKeyboardState(lastAction).keys);
            releasedKeys.addAll(getKeyboardState(lastAction).keys);
            releasedKeys.removeAll(state.keys);
        }
        for (String key : newlyHitKeys) {
            InputMappings.Input input = InputMappings.getInputByName(key);
            if (lastAction != null &&
                    mc.currentScreen == null &&
                    key.equals("key.keyboard.escape") && mc.gameSettings.envPort != 0 ) {
                // prevent from sending "pause" menu command when in the environment
                continue;
            }
            mc.keyboardListener.onKeyEventImpl(input.getKeyCode(), 0, 1, mods);
        }
        for (String key : releasedKeys) {
            InputMappings.Input input = InputMappings.getInputByName(key);
            mc.keyboardListener.onKeyEventImpl(input.getKeyCode(), 0, 0, mods);
        }
        for (Character c : state.chars.toCharArray()) {
            mc.keyboardListener.onCharEventImpl(c.charValue(), mods);
        }
        mc.keyboardListener.getState();
    }

    private void execMouseScroll(double dwheel) {
        if (dwheel != 0.0) {
            Minecraft mc = Minecraft.getInstance();
            mc.mouseHelper.scrollCallbackImpl(mc.getMainWindow().getHandle(), 0, dwheel);
        }
    }

    public Mode getMode() {return mode; }

    public void loadWorldFromZip(String zipFile) {
        Minecraft mc = Minecraft.getInstance();
        List<String> zipEntries = ZipUtil.listZip(zipFile);
        String tmpDir = System.getProperty("java.io.tmpdir");
        Path saveBasePath = Paths.get(tmpDir, RandomHelper.getRandomHexString());
        System.out.println("Unzipping " + zipFile + " -> " + saveBasePath);
        // TODO maybe clean the temp save folder
        ZipUtil.unzip(zipFile, saveBasePath.toString());
        // zip folder structure is as follows:
        // . -> saves -> New World Name -> <actual save data>
        // thus, the world can be obtaines as below:
        String saveName = zipEntries.get(0).split("/")[2];
        // and, after the unzipping, we need to add "saves" to the basePath to
        // obtain the correct folder
        mc.loadWorld(saveBasePath.resolve("saves"), saveName);
    }

    public void loadWorldAndReplay(String prefix) {
        // TODO handle file that is not at the beginning of the episode (e.g. tick counter of 6003)
        loadWorldFromZip(prefix + ".zip");
        sendFromFile(prefix + ".jsonl", 0);
    }
}
