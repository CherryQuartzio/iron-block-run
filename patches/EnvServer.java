package com.minerl.multiagent.env;

import com.google.common.base.Charsets;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.microsoft.Malmo.Schemas.*;
import com.microsoft.Malmo.Utils.JSONWorldDataHelper;
import com.minerl.multiagent.RandomHelper;
import com.minerl.multiagent.recorder.AzureUpload;
import com.minerl.multiagent.recorder.PlayRecorder;
import net.minecraft.client.*;
import net.minecraft.client.entity.player.ClientPlayerEntity;
import net.minecraft.client.gui.screen.ConnectingScreen;
import net.minecraft.client.gui.screen.MainMenuScreen;
import net.minecraft.client.multiplayer.ServerAddress;
import net.minecraft.client.multiplayer.ServerData;
import net.minecraft.entity.Entity;
import net.minecraft.entity.EntityType;
import net.minecraft.entity.SpawnReason;
import net.minecraft.entity.ai.attributes.Attributes;
import net.minecraft.entity.passive.horse.AbstractHorseEntity;
import net.minecraft.entity.passive.horse.HorseEntity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.item.ItemStack;
import net.minecraft.item.Items;
import net.minecraft.nbt.CompoundNBT;
import net.minecraft.util.math.BlockPos;
import net.minecraft.entity.player.PlayerInventory;
import net.minecraft.inventory.EquipmentSlotType;
import net.minecraft.item.Item;
import net.minecraft.item.ItemStack;
import net.minecraft.profiler.IResultableProfiler;
import net.minecraft.server.MinecraftServer;
import net.minecraft.util.ResourceLocation;
import net.minecraft.util.SoundCategory;
import net.minecraft.util.datafix.codec.DatapackCodec;
import net.minecraft.util.registry.DynamicRegistries;
import net.minecraft.util.registry.Registry;
import net.minecraft.util.SoundCategory;
import net.minecraft.world.Difficulty;
import net.minecraft.world.GameRules;
import net.minecraft.world.GameType;
import net.minecraft.world.World;
import net.minecraft.world.WorldSettings;
import net.minecraft.world.server.ServerWorld;
import net.minecraft.world.gen.settings.DimensionGeneratorSettings;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.jmx.Server;

import javax.xml.bind.JAXBElement;
import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.Charset;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.Optional;

public class EnvServer {
    private static Logger LOGGER = LogManager.getLogger();
    private static String hello = "<MalmoEnv" ;
    private static final int stepClientTagLength = "<StepClient_>".length();
    private static final int stepServerTagLength = "<StepServer_>".length();
    private boolean iwanttoquit = false;
    private boolean doneOnDeath = false;

    static final int BYTES_INT = 4;
    static final int BYTES_DOUBLE = 8;
    // this many steps with noop action will be taken at the beginning of
    // the episode. Helps to render scene more fully and avoid unrendered chunks
    // TODO peterz validate this is actually still necessary, given the sync chunk loading
    private static final int DEFAULT_SKIP_FIRST_FRAMES = 20;

    // Race horse stats (formerly DrawEntity NBTData in agent mission XML).
    private static final double HORSE_MOVEMENT_SPEED = 0.2;
    private static final double HORSE_JUMP_STRENGTH = 0.85;
    private static final double HORSE_MAX_HEALTH = 20.0;
    private static final float HORSE_HEALTH = 20.0F;
    private static final int HORSE_VARIANT = 1029;

    private int envTickCounter = -1;
    private MissionInit missionInit;

    private int port;
    private String version;
    public EnvServer(int port, String version) {
        this.port = port;
        this.version = version;
    }

    public void serve() {
        ServerSocket serverSocket;
        try {
            serverSocket = new ServerSocket(port);
            serverSocket.setPerformancePreferences(0, 2, 1);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        // expected malmo text
        System.out.println("***** Start MalmoEnvServer on port " + port);
        System.out.println("CLIENT enter state: DORMANT");
        System.out.println("SERVER enter state: DORMANT");

        while (!iwanttoquit) {
            try {
                final Socket socket = serverSocket.accept();
                socket.setTcpNoDelay(true);
                Thread thread = new Thread("EnvServerSocketHandler") {
                    public void run() {
                        boolean running = false;
                        try {
                            checkHello(socket);

                            while (true) {

                                DataInputStream din = new DataInputStream(socket.getInputStream());
                                int hdr = 0;
                                try {
                                    hdr = din.readInt();
                                } catch (EOFException e) {
                                    LOGGER.debug("Incoming socket connection closed, likely by peer (without Exit message): " + e);
                                    socket.close();
                                    break;
                                }
                                byte[] data = new byte[hdr];

                                din.readFully(data);
                                String command = new String(data, Charset.forName("UTF-8"));

                                // TODO this comms schema is seriously an atrocity
                                // Needs to be rewritten such that schema is explicit
                                // maybe use grpc or something like that?
                                if (command.startsWith("<StepClient")) {

                                    stepClient(command, socket, din);

                                } else if (command.startsWith("<StepServer")) {

                                    stepServer(command, socket);

                                } else if (command.startsWith("<Peek")) {

                                    peek(command, socket, din);

                                } else if (command.startsWith("<MissionInit")) {

                                    if (initMission(din, command, socket)) {
                                        running = true;
                                    }

                                } else if (command.startsWith("<Quit")) {

                                    quit(command, socket);

                                    // profiler.profilingEnabled = false;

                                } else if (command.startsWith("<Exit")) {

                                    quit(command, socket);
                                    AzureUpload.finish();
                                    Minecraft.getInstance().shutdown();

                                    // profiler.profilingEnabled = false;

                                    return; // exit

                                } else if (command.startsWith("<Close")) {

                                    // close(command, socket);
                                    // profiler.profilingEnabled = false;

                                }  else if (command.startsWith("<Echo")) {
                                    command = "<Echo>" + command + "</Echo>";
                                    data = command.getBytes(Charset.forName("UTF-8"));
                                    hdr = data.length;

                                    DataOutputStream dout = new DataOutputStream(socket.getOutputStream());
                                    dout.writeInt(hdr);
                                    dout.write(data, 0, hdr);
                                    dout.flush();
                                } else if (command.startsWith("<Disconnect")) {
                                    socket.close();
                                    break;
                                } else {
                                    throw new IOException("Unknown env service command: " + command);
                                }
                            }
                        } catch (IOException ioe) {
                            ioe.printStackTrace();
                            LOGGER.fatal("MalmoEnv socket error: " + ioe + " (can be on disconnect)");

                            // TimeHelper.SyncManager.debugLog("[MALMO_ENV_SERVER] MalmoEnv socket error");
                            try {
                                if (running) {
                                    LOGGER.info("Want to quit on disconnect.");
                                    System.out.println( "[LOGTOPY] " + "Want to quit on disconnect.");
                                    setWantToQuit();
                                }
                                socket.close();
                            } catch (IOException ioe2) {
                            }
                        } catch (Exception e) {
                            LOGGER.error("Error while processing commands", e);
                            try {
                                socket.close();
                            } catch (IOException ioe2) {
                            }
                        }
                    }
                };
                thread.start();
            } catch (IOException ioe) {
                LOGGER.log(Level.FATAL, "MalmoEnv service exits on " + ioe);
                LOGGER.error("IO Error while processing commands", ioe);
            } catch (Exception e) {
                LOGGER.error("Error while processing commands", e);
            }
        }
    }

    private void checkHello(Socket socket) throws IOException {

        DataInputStream din = new DataInputStream(socket.getInputStream());
        int hdr = din.readInt();
        if (hdr <= 0 || hdr > hello.length() + 8) // Version number may be somewhat longer in future.
            throw new IOException("Invalid MalmoEnv hello header length");
        byte[] data = new byte[hdr];
        din.readFully(data);
        if (!new String(data).startsWith(hello + version))
            throw new IOException("MalmoEnv invalid protocol or version - expected " + hello + version);

    }

    private void setWantToQuit() {
        // todo make sure this is really neccessary
        iwanttoquit = true;
    }


    boolean initMission(DataInputStream din, String command, Socket socket) throws IOException, InterruptedException {
        int hdr;
        byte[] data;
        hdr = din.readInt();
        data = new byte[hdr];
        din.readFully(data);
        String id = new String(data, Charsets.UTF_8);
        LOGGER.info("Received Mission token " + id);
        LOGGER.info("Received mission init command  " + command);

        // todo world settings and dimension generator settings from mission xml
        Minecraft mc = Minecraft.getInstance();
        missionInit = MissionSpec.decodeMissionInit(command);

        // Manual parsing seed from the token, as done in older code
        // id is string of ":" separated values. The sixth is seed if it exists.
        // This is done to support the `env.seed` command of Gym-like environments, which
        // would change/modify the mission XML constantly.
        // WorldSeed handler is also supported below, but this overrides WorldSeed.

        Long seed = null;
        String[] parts = id.split(":");
        if (parts.length >= 6) {
            try {
                seed = Long.parseLong(parts[5]);
            } catch (NumberFormatException e) {
                LOGGER.error("Received invalid seed: " + parts[5]);
            }
        }
        if (seed == null) {
            // If seed was not set in mission token, see if XML file has it
            seed = getSeed(missionInit);
        }

        final Long final_seed = seed;

        setGameSetttings(missionInit);
        mc.getSession().setUsername(missionInit.getMission().getAgentSection().get(0).getName());
        setUsername(missionInit);
        mc.execute(() -> loadOrCreateWorld(missionInit, final_seed));
        while (!PlayRecorder.getInstance().isRecording()) {
            Thread.sleep(10);
        }

        mc.execute(() -> setAgentInventory(mc.player, missionInit));
        mc.execute(() -> {
            setAgentPosition(mc.player, missionInit);
            enforceAgentGameMode(missionInit);
        });
        envTickCounter = PlayRecorder.getInstance().getTickCounter();
        // TODO possibly remove
        // if necessary, can set this from missionInit ?
        int skipFrames = DEFAULT_SKIP_FIRST_FRAMES;
        for (int i = 0; i < skipFrames; i++) {
            execActions("camera 0 0.0", 0);
            waitForNextObservation();
        }

        mc.execute(() -> applyWorldDecorators(missionInit));
        for (int i = 0; i < 5; i++) {
            execActions("camera 0 0.0", 0);
            waitForNextObservation();
        }

        mc.execute( () -> {
                    Pos startV = getAgentStart(missionInit).getVelocity();
                    if (startV != null) {
                        mc.player.setMotion(startV.getX(), startV.getY(), startV.getZ());
                    }
                });


        DataOutputStream dout = new DataOutputStream(socket.getOutputStream());
        dout.writeInt(4);
        dout.writeInt(1);
        dout.flush();
        return true;
    }

    private void setAgentInventory(ClientPlayerEntity player, MissionInit missionInit) {
        // using forEach and lambda instead of for loop to avoid atrociously long
        // type name
        AgentStart.Inventory inventory = getAgentStart(missionInit).getInventory();
        if (inventory == null) {
            return;
        }
        inventory.getInventoryObject().forEach( e -> {
            String type = e.getValue().getType();
            int quantity = e.getValue().getQuantity();
            int slot = e.getValue().getSlot();
            Item item = Registry.ITEM.getOrDefault(new ResourceLocation(type));
            player.inventory.setInventorySlotContents(slot, new ItemStack(item, quantity));
            Minecraft.getInstance().getIntegratedServer().getPlayerList().getPlayers().forEach( p -> {
                    p.inventory.setInventorySlotContents(slot, new ItemStack(item, quantity));
                });
        });
    }

    private void setAgentPosition(ClientPlayerEntity player, MissionInit missionInit) {
        PosAndDirection startPos = getAgentStart(missionInit).getPlacement();
        if (startPos == null) {
            return;
        }
        player.setPosition(startPos.getX(), startPos.getY(), startPos.getZ());
        player.rotationYaw = startPos.getYaw();
        player.rotationPitch = startPos.getPitch();
    }

    private void enforceAgentGameMode(MissionInit missionInit) {
        Minecraft mc = Minecraft.getInstance();
        if (mc.player == null) {
            return;
        }
        AgentSection section = missionInit.getMission().getAgentSection().get(0);
        GameType mode = GameType.getByName(section.getMode().name().toLowerCase());
        mc.player.setGameType(mode);
        if (mc.playerController != null) {
            mc.playerController.setGameType(mode);
        }
        mc.player.abilities.isFlying = false;
        mc.player.abilities.allowFlying = false;
        mc.player.sendPlayerAbilities();
    }

    private void applyWorldDecorators(MissionInit missionInit) {
        Minecraft mc = Minecraft.getInstance();
        World world = mc.world;
        if (world == null) {
            LOGGER.warn("Cannot apply world decorators: world is null");
            return;
        }
        ServerHandlers serverHandlers = missionInit.getMission().getServerSection().getServerHandlers();
        if (serverHandlers == null) {
            return;
        }
        for (Object decorator : serverHandlers.getWorldDecorators()) {
            if (!(decorator instanceof DrawingDecorator)) {
                continue;
            }
            DrawingDecorator drawing = (DrawingDecorator) decorator;
            for (JAXBElement<? extends DrawObjectType> element : drawing.getDrawObjectType()) {
                if (element.getValue() instanceof DrawEntity) {
                    try {
                        spawnDrawEntity((DrawEntity) element.getValue(), world, mc);
                    } catch (Exception e) {
                        LOGGER.error("Error spawning DrawEntity", e);
                    }
                }
            }
        }
    }

    private Optional<EntityType<?>> resolveEntityType(String entityName) {
        ResourceLocation entityId = ResourceLocation.tryCreate(entityName);
        Optional<EntityType<?>> optionalType = Registry.ENTITY_TYPE.getOptional(entityId);
        if (optionalType.isPresent()) {
            return optionalType;
        }
        if (!entityName.contains(":")) {
            optionalType = Registry.ENTITY_TYPE.getOptional(new ResourceLocation("minecraft", entityName.toLowerCase()));
        }
        return optionalType;
    }

    private void spawnDrawEntity(DrawEntity drawEntity, World world, Minecraft mc) throws Exception {
        String entityName = drawEntity.getType().getValue();
        Optional<EntityType<?>> optionalType = resolveEntityType(entityName);
        if (!optionalType.isPresent()) {
            LOGGER.warn("Unknown entity type: " + entityName);
            return;
        }

        MinecraftServer server = mc.getIntegratedServer();
        if (server == null) {
            LOGGER.warn("Cannot spawn entity without integrated server");
            return;
        }
        ServerWorld serverWorld = server.getWorld(World.OVERWORLD);

        double x = drawEntity.getX().doubleValue();
        double y = drawEntity.getY().doubleValue();
        double z = drawEntity.getZ().doubleValue();
        float yaw = drawEntity.getYaw() != null ? drawEntity.getYaw().floatValue() : 0f;
        BlockPos spawnPos = new BlockPos(x, y, z);

        EntityType<?> entityType = optionalType.get();
        Entity entity;
        if (entityType == EntityType.HORSE) {
            entity = EntityType.HORSE.spawn(
                serverWorld,
                (CompoundNBT) null,
                (net.minecraft.util.text.ITextComponent) null,
                mc.player,
                spawnPos,
                SpawnReason.COMMAND,
                true,
                false
            );
        } else {
            entity = entityType.create(serverWorld);
            if (entity == null) {
                LOGGER.warn("Could not create entity for type: " + entityName);
                return;
            }
            entity.setLocationAndAngles(x, y, z, yaw, 0);
        }

        if (entity == null) {
            LOGGER.warn("Could not spawn entity for type: " + entityName);
            return;
        }

        if (entity instanceof AbstractHorseEntity) {
            configureSpawnedHorse((AbstractHorseEntity) entity, mc, x, y, z, yaw);
        }

        serverWorld.getBlockState(spawnPos);
        if (entity instanceof AbstractHorseEntity) {
            ((AbstractHorseEntity) entity).setPositionAndUpdate(x, y, z);
        }
    }

    private void configureSpawnedHorse(
        AbstractHorseEntity horse,
        Minecraft mc,
        double x,
        double y,
        double z,
        float yaw
    ) {
        horse.setLocationAndAngles(x, y, z, yaw, 0);
        horse.setHorseTamed(true);
        if (mc.player != null) {
            horse.setOwnerUniqueId(mc.player.getUniqueID());
        }
        horse.func_230266_a_(SoundCategory.PLAYERS);

        horse.getAttribute(Attributes.MOVEMENT_SPEED).setBaseValue(HORSE_MOVEMENT_SPEED);
        horse.getAttribute(Attributes.HORSE_JUMP_STRENGTH).setBaseValue(HORSE_JUMP_STRENGTH);
        horse.getAttribute(Attributes.MAX_HEALTH).setBaseValue(HORSE_MAX_HEALTH);
        horse.setHealth(HORSE_HEALTH);

        if (horse instanceof HorseEntity) {
            CompoundNBT variantNbt = new CompoundNBT();
            variantNbt.putInt("Variant", HORSE_VARIANT);
            ((HorseEntity) horse).readAdditional(variantNbt);
        }
    }

    private String getSaveFile(MissionInit missionInit) {
        return missionInit.getMission().getAgentSection().get(0).getAgentStart().getLoadWorldFile();
    }

    private void setUsername(MissionInit missionInit) {
        String username = getAgentStart(missionInit).getMultiplayerUsername();
        if (username != null) {
            Minecraft.getInstance().getSession().setUsername(username);
        }
    }
    
    private void setGameSetttings(MissionInit missionInit) {
        Minecraft mc = Minecraft.getInstance();
        GameSettings settings = mc.gameSettings;
        AgentStart agentStart = getAgentStart(missionInit);
        settings.gamma = agentStart.getGammaSetting();
        settings.fov = agentStart.getFOVSetting();
        settings.disableRecorder = agentStart.isEnableRecorder() == null || !agentStart.isEnableRecorder();
        settings.fakeCursorSize = agentStart.getFakeCursorSize();
        float guiScale = agentStart.getGuiScale();
        settings.setSoundLevel(SoundCategory.MASTER, 0.0f);

        MainWindow window = mc.getMainWindow();
        getAgentHandlers().filter(h -> h instanceof VideoProducer).forEach(h -> {
            VideoProducer vp = (VideoProducer)h;
            System.out.println("Setting width, height to " + vp.getWidth() + ", " + vp.getHeight());
            double fbToWindowRatio = (double) window.getFramebufferWidth() / window.getWidth();
            mc.execute(() -> {
                window.resize((int) (vp.getWidth() / fbToWindowRatio), (int) (vp.getHeight() / fbToWindowRatio));
                mc.updateWindowSize();
                window.setGuiScale(guiScale);
            });
        });

        System.out.println("Gamma: " + settings.gamma);
        System.out.println("FOV: " + settings.fov);
        System.out.println("GuiScale: " + guiScale);
    }

    private AgentStart getAgentStart(MissionInit missionInit) {
        return missionInit.getMission().getAgentSection().get(0).getAgentStart();
    }

    private void loadOrCreateWorld(MissionInit missionInit, Long seed) {
        String saveZipFile = getSaveFile(missionInit);
        if (saveZipFile == null) {
            String serverAddress = getServerAddress(missionInit);
            if (serverAddress == null) {
                createNewWorld(missionInit, seed);
            } else {
                connectToServer(serverAddress);
            }
        } else {
            ReplaySender.getInstance().loadWorldFromZip(saveZipFile);
        }
    }

    private String getServerAddress(MissionInit missionInit) {
        return getServerInit(missionInit).getRemoteServer();
    }

    private void connectToServer(String serverAddress) {
        Minecraft mc = Minecraft.getInstance();
        ServerData serverData = new ServerData("social", serverAddress, true);
        ServerAddress serveraddress = ServerAddress.fromString(serverData.serverIP);
        mc.displayGuiScreen(new ConnectingScreen(new MainMenuScreen(false), mc, serverData));
    }

    private void createNewWorld(MissionInit missionInit) {
        createNewWorld(missionInit, null);
    }

    private void createNewWorld(MissionInit missionInit, Long seed) {
        Minecraft mc = Minecraft.getInstance();
        boolean bonusChest = isBonusChest(missionInit);
        boolean generateFeatures = isGenerateFeatures(missionInit);
        boolean spawnInVillage = isSpawnInVillage(missionInit);
        this.doneOnDeath = isDoneOnDeath(missionInit);
        if (this.doneOnDeath) {
            // If we are resetting environment, this ensures
            // the flag is reset to false
            mc.setHasPlayerRespawned(false);
        }
        if (seed == null) {
            seed = new Random().nextLong();
            System.out.println("Seed not provided, generating random one: " + String.valueOf(seed));
        }
        String worldName = "mcpworld" + RandomHelper.getRandomHexString();
        String spawnBiome = getAgentStart(missionInit).getPreferredSpawnBiome();
        if (spawnBiome != null) {
            checkValidBiome(spawnBiome);
            MinecraftServer.setSpawnBiomePredicate( b -> b.getCategory().getName().equals(spawnBiome) );
        }

        if (spawnInVillage) {
            MinecraftServer.setSpawnInVillage(true);
        }


        WorldSettings worldSettings = new WorldSettings(worldName, GameType.SURVIVAL, false, Difficulty.HARD, false, new GameRules(), DatapackCodec.VANILLA_CODEC);
        DimensionGeneratorSettings dms = DimensionGeneratorSettings.fromDynamicRegistries(DynamicRegistries.getImpl(), seed, generateFeatures, bonusChest);
        mc.createWorld(worldName, worldSettings, DynamicRegistries.getImpl(), dms);
    }

    private void checkValidBiome(String spawnBiome) {
        Set<String> biomeCategories = DynamicRegistries.getImpl().getRegistry(Registry.BIOME_KEY).getEntries().stream()
                .map(e -> e.getValue().getCategory().getName())
                .collect(Collectors.toSet());
        if (!biomeCategories.contains(spawnBiome)) {
            LOGGER.error("Bad starting biome " + spawnBiome);
            LOGGER.error("Biome should be one of the following: ");
            for (String b : biomeCategories) {
                LOGGER.error("- " + b);
            }
            throw new RuntimeException("Bad starting biome " + spawnBiome);
        }
    }

    private Long getSeed(MissionInit missionInit){
        // return null;
        return getAgentStart(missionInit).getWorldSeed();
    }

    private boolean isBonusChest(MissionInit missionInit) {
        Boolean bonusChest = getAgentStart(missionInit).isBonusChest();
        return bonusChest != null && bonusChest;
    }

    private boolean isGenerateFeatures(MissionInit missionInit) {
        Boolean genFeatures = getAgentStart(missionInit).isGenerateFeatures();
        return genFeatures == null || genFeatures;
    }

    private boolean isSpawnInVillage(MissionInit missionInit) {
        Boolean spawnInVillage = getAgentStart(missionInit).isSpawnInVillage();
        return spawnInVillage != null && spawnInVillage;
    }

    private boolean isDoneOnDeath(MissionInit missionInit) {
        Boolean doneOnDeath = getAgentStart(missionInit).isDoneOnDeath();
        return doneOnDeath != null && doneOnDeath;
    }


    void peek(String command, Socket socket, DataInputStream din) throws IOException, ExecutionException, InterruptedException {
        Minecraft mc = Minecraft.getInstance();
        DataOutputStream dout = new DataOutputStream(socket.getOutputStream());
        byte[] obs = getPOVObservation();
        boolean done = false;
        String info = getInfo();
        dout.writeInt(obs.length);
        dout.write(obs);
        byte[] infoBytes = info.getBytes(Charset.forName("UTF-8"));
        dout.writeInt(infoBytes.length);
        dout.write(infoBytes);
        dout.writeInt(1);
        dout.writeByte(done ? 1 : 0);
        dout.flush();
    }

    private byte[] getPOVObservation() {
        return PlayRecorder.getInstance().getLastImageBytes();
    }

    private void waitForNextObservation() {
        // this dependency on tick counter seems a little spaghetti
        // ideally, instead addAction returns a future on next observation (gym-style)
        // these futures are then resolved by ReplaySender or similar entity
        PlayRecorder pr = PlayRecorder.getInstance();

        try {
            synchronized (pr) {
                while (envTickCounter == pr.getTickCounter()) {
                    pr.wait();
                }
            }
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        envTickCounter = PlayRecorder.getInstance().getTickCounter();
    }


    private void stepClient(String command, Socket socket, DataInputStream din) throws IOException {
        Minecraft mc = Minecraft.getInstance();
        String actions = command.substring(stepClientTagLength, command.length() - (stepClientTagLength + 2));
        int options =  Character.getNumericValue(command.charAt(stepServerTagLength - 2));
        boolean withInfo = options == 0 || options == 2;
        envTickCounter = PlayRecorder.getInstance().getTickCounter();
        execActions(actions, options);
        waitForNextObservation();
        byte[] obs = getPOVObservation();
        boolean done = mc.player == null || !mc.player.isAlive() || (this.doneOnDeath && mc.isHasPlayerRespawned());
        boolean sent = true;
        DataOutputStream dout = new DataOutputStream(socket.getOutputStream());
        dout.writeInt(obs.length);
        dout.write(obs);
        dout.writeInt(BYTES_DOUBLE + 2);
        dout.writeDouble(0.0);
        dout.writeByte(done ? 1 : 0);
        dout.writeByte(sent ? 1 : 0);

        if (withInfo) {
            String info = getInfo();
            byte[] infoBytes = info.getBytes(Charsets.UTF_8);
            dout.writeInt(infoBytes.length);
            dout.write(infoBytes);
        }
        dout.flush();
    }

    private Stream<Object> getAgentHandlers() {
        return missionInit.getMission().getAgentSection().get(0).getAgentHandlers().getAgentMissionHandlers().stream();
    }
    
    private static ServerInitialConditions getServerInit(MissionInit missionInit) {
        return missionInit.getMission().getServerSection().getServerInitialConditions();
    }

    private String getInfo() {
        Minecraft mc = Minecraft.getInstance();
        JsonObject infoJson = new JsonObject();
        List<Object> handlers = missionInit.getMission().getAgentSection().get(0).getAgentHandlers().getAgentMissionHandlers();
        if (mc.player != null) {
            getAgentHandlers().filter(h -> h instanceof ObservationFromFullInventory).limit(1)
                    .forEach(h -> infoJson.add("inventory", getInventoryJson()));
            getAgentHandlers().filter(h -> h instanceof ObservationFromFullStats).limit(1)
                    .forEach(h -> {
                        JSONWorldDataHelper.buildAllStats(infoJson, mc.player);
                    });
            getAgentHandlers().filter(h -> h instanceof ObservationFromEquippedItem).limit(1)
		    .forEach(h -> {
		        infoJson.add("equipped_items", getEquippedItemJson());
		    });
        }

        infoJson.addProperty("isGuiOpen", mc.currentScreen != null);
        return infoJson.toString();
    }

    public static JsonArray getInventoryJson() {
        JsonArray result = new JsonArray();
        PlayerInventory inventory = Minecraft.getInstance().player.inventory;
        for (ItemStack is: inventory.mainInventory) {
            if (is.getCount() > 0) {
                JsonObject stack = new JsonObject();
                stack.addProperty("type", is.getItem().toString());
                stack.addProperty("quantity", is.getCount());
                result.add(stack);
            }
        }
        return result;
    }
    
    public static JsonObject getEquippedItemJson() {
        JsonObject result = new JsonObject();
        PlayerEntity player = Minecraft.getInstance().player;
        assert player != null;
        result.addProperty("mainhand", getEquipmentJsonObjectFromPlayer(player, EquipmentSlotType.MAINHAND));
        result.addProperty("offhand", getEquipmentJsonObjectFromPlayer(player, EquipmentSlotType.OFFHAND));
        result.addProperty("head", getEquipmentJsonObjectFromPlayer(player, EquipmentSlotType.HEAD));
        result.addProperty("chest", getEquipmentJsonObjectFromPlayer(player, EquipmentSlotType.CHEST));
        result.addProperty("legs", getEquipmentJsonObjectFromPlayer(player, EquipmentSlotType.LEGS));
        result.addProperty("feet", getEquipmentJsonObjectFromPlayer(player, EquipmentSlotType.FEET));
        return result;
    }

    private static String getEquipmentJsonObjectFromPlayer(PlayerEntity player, EquipmentSlotType type) {
        JsonObject result = new JsonObject();
        ItemStack item = player.getItemStackFromSlot(type);
        result.addProperty("type", item.getItem().toString());
        result.addProperty("maxDamage", item.getMaxDamage());
        result.addProperty("damage", item.getDamage());
        return result.toString();
    }

    public static void execActions(String actions, int options) {
        KeyboardListener.State keysState = constructKeyboardState(actions);
        MouseHelper.State mouseState = constructMouseState(actions);
        PlayRecorder.getInstance().setMouseKeyboardState(mouseState, keysState);
        ReplaySender.getInstance().addAction(mouseState, keysState);
    }

    private static KeyboardListener.State constructKeyboardState(String actions) {
        List<String> keysPressed = new ArrayList<>();
        for (String action: actions.split("\n")) {
           String[] splitAction = action.trim().split(" ");
           if (!splitAction[0].equals("camera") && !splitAction[0].equals("dwheel")) {
               if (splitAction.length > 1 && Integer.parseInt(splitAction[1]) == 1) {
                   String key = actionToKey(splitAction[0]);
                   if (key != null) {
                       keysPressed.add(key);
                   }
               }
           }
        }
        return new KeyboardListener.State(keysPressed, Collections.emptyList(), "");
    }

    private static MouseHelper.State constructMouseState(String actions) {
        List<Integer> buttonsPressed = new ArrayList<>();
        double dx = 0;
        double dy = 0;
        double dwheel = 0;
        // 2400 is mouse dx that corresponds to a full (360 degree) turn, hence the
        // formula below to compute mouse -> camera sensitivity
        // the value is screen resolution independent, as it turns out
        // there is a manual test in monorepo (minecraft/tests/test_turn.py) that can be used to
        // validate that, indeed, with this value of sensitivity, 360 degrees in camera pitch make
        // a full turn, and 90 degrees in yaw make agent look fully up or fully down
        double sensitivity = 2400.0 / 360;
        for (String action: actions.split("\n")) {
            String[] splitAction = action.trim().split(" ");
            if (splitAction[0].equals("camera")) {
                dx = Double.parseDouble(splitAction[2]) * sensitivity;
                dy = Double.parseDouble(splitAction[1]) * sensitivity;
            } else if (splitAction[0].equals("dwheel")) {
                dwheel = Double.parseDouble(splitAction[1]);
            } else {
                if (splitAction.length > 1 && Integer.parseInt(splitAction[1]) == 1) {
                    Integer key = actionToMouseButton(splitAction[0]);
                    if (key != null) {
                        buttonsPressed.add(key);
                    }
                }
            }
        }
        return new MouseHelper.State(0, 0, dx, dy, dwheel, buttonsPressed, Collections.emptyList());
    }

    private static Integer actionToMouseButton(String action) {
        if (action.equals("attack")) {
            return 0;
        } else if (action.equals("use")) {
            return 1;
        } else if (action.equals("pickItem")) {
            return 2;
        }
        return null;
    }

    private static String actionToKey(String action) {
        if (action.equals("forward")) {
            return "key.keyboard.w";
        } else if (action.equals("back")) {
            return "key.keyboard.s";
        } else if (action.equals("left")) {
            return "key.keyboard.a";
        } else if (action.equals("right")) {
            return "key.keyboard.d";
        } else if (action.equals("jump")) {
            return "key.keyboard.space";
        } else if (action.equals("sprint")) {
            return "key.keyboard.left.control";
        } else if (action.equals("sneak")) {
            return "key.keyboard.left.shift";
        } else if (action.startsWith("hotbar")) {
            return "key.keyboard." + action.split("\\.")[1];
        } else if (action.equals("inventory")) {
            return "key.keyboard.e";
        } else if (action.equals("drop")) {
            return "key.keyboard.q";
        } else if (action.equals("swapHands")) {
            return "key.keyboard.f";
        } else if (action.equals("ESC")) {
            return "key.keyboard.escape";
        }
        return null;
    };

    void stepServer(String command, Socket socket) {
        // step server
    }

    // Handler for <Quit> (quit mission) messages.
    private void quit(String command, Socket socket) throws IOException, InterruptedException {
        Minecraft mc = Minecraft.getInstance();
        if (mc.getProfiler() instanceof IResultableProfiler) {
            File profileDump = new File("profile-results-" + (new SimpleDateFormat("yyyy-MM-dd_HH.mm.ss")).format(new Date()) + ".txt");
            ((IResultableProfiler)mc.getProfiler()).getResults().writeToFile(profileDump.getAbsoluteFile());
        }
        PlayRecorder.getInstance().finishAndResetEpisode();
        ReplaySender.getInstance().stop();

        while (!(mc.currentScreen instanceof MainMenuScreen)) {
            Thread.sleep(10);
        }
        DataOutputStream dout = new DataOutputStream(socket.getOutputStream());
        dout.writeInt(4);
        dout.writeInt(1);
        dout.flush();
    }
}
