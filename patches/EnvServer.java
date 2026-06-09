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
import net.minecraft.entity.player.ServerPlayerEntity;
import net.minecraft.item.ItemStack;
import net.minecraft.item.Items;
import net.minecraft.nbt.CompoundNBT;
import net.minecraft.util.math.AxisAlignedBB;
import net.minecraft.util.math.BlockPos;
import net.minecraft.entity.player.PlayerInventory;
import net.minecraft.inventory.EquipmentSlotType;
import net.minecraft.item.Item;
import net.minecraft.item.ItemStack;
import net.minecraft.profiler.IResultableProfiler;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.integrated.IntegratedServer;
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
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BooleanSupplier;
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
    // Tracks horse-mount state across steps so we can log when (and why) the
    // player gets dismounted. Reset at the start of each mission.
    private boolean wasRiding = false;

    static final int BYTES_INT = 4;
    static final int BYTES_DOUBLE = 8;
    // this many steps with noop action will be taken at the beginning of
    // the episode. Helps to render scene more fully and avoid unrendered chunks
    // TODO peterz validate this is actually still necessary, given the sync chunk loading
    private static final int DEFAULT_SKIP_FIRST_FRAMES = 20;
    private static final long GAME_THREAD_TIMEOUT_MS = 5000L;
    private static final long GAME_THREAD_TASK_TIMEOUT_MS = 10000L;
    private static final long RESET_TASK_TIMEOUT_MS = 60000L;
    private static final long RECORDING_WAIT_TIMEOUT_MS = 30000L;
    private static final long MAIN_MENU_WAIT_TIMEOUT_MS = 30000L;
    private static final long WORLD_READY_TIMEOUT_MS = 60000L;
    private static final long WORLD_LOAD_TIMEOUT_MS = 120000L;
    private static final long OBSERVATION_WAIT_TIMEOUT_MS = 30000L;

    // Race horse stats (formerly DrawEntity NBTData in agent mission XML).
    private static final double HORSE_MOVEMENT_SPEED = 0.2;
    private static final double HORSE_JUMP_STRENGTH = 0.85;
    private static final double HORSE_MAX_HEALTH = 20.0;
    private static final float HORSE_HEALTH = 20.0F;
    private static final int HORSE_VARIANT = 1029;
    // Horizontal half-extent (blocks) for per-episode horse cleanup. Must stay
    // bounded — a world-spanning AABB hangs the game thread (chunk-section walk).
    private static final double HORSE_CLEANUP_RADIUS = 2048.0;

    private int envTickCounter = -1;
    private MissionInit missionInit;
    private String activeWorldSource = null;
    private boolean lanPublished = false;
    private volatile boolean integratedServerAlive = false;
    private boolean configLogged = false;

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

                                    quit(command, socket, false);

                                    // profiler.profilingEnabled = false;

                                } else if (command.startsWith("<Exit")) {

                                    quit(command, socket, true);
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
        wasRiding = false;  // reset mount tracking for the new episode

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

        this.doneOnDeath = isDoneOnDeath(missionInit);

        logPersistentConfig(mc);

        setGameSetttings(missionInit);
        mc.getSession().setUsername(missionInit.getMission().getAgentSection().get(0).getName());
        setUsername(missionInit);

        String worldSrc = getSaveFile(missionInit);
        boolean reuseWorld = canReuseWorld(worldSrc);

        // NOTE: never block the socket thread waiting on a game-thread task here.
        // Once recording starts, ReplaySender parks the game thread in tick() until
        // an action arrives, and the executor queue only drains between ticks. The
        // skip-frame loop below feeds one action per tick, which is what advances
        // ticks AND drains any queued reset task. This mirrors stock MineRL.
        if (reuseWorld) {
            LOGGER.info("[Persistent] Soft episode reset — reusing loaded world");
            // Recording + integrated server stay alive across the soft reset.
            mc.execute(() -> {
                resetAgentForNewEpisode(missionInit, true);
                PlayRecorder.getInstance().softResetEpisode();
                ReplaySender.getInstance().clearEpisodeState();
            });
        } else {
            integratedServerAlive = false;
            activeWorldSource = null;
            lanPublished = false;
            mc.execute(() -> loadOrCreateWorld(missionInit, final_seed));
            activeWorldSource = worldSrc;
            // World is loading with ReplaySender OFF, so the game thread ticks
            // freely and PlayRecorder starts recording on its own. Plain poll —
            // no pumping needed (and pumping is a no-op while mode == OFF).
            waitForRecording();
            mc.execute(() -> resetAgentForNewEpisode(missionInit, false));
        }

        integratedServerAlive = true;
        maybeOpenToLan();

        envTickCounter = PlayRecorder.getInstance().getTickCounter();
        int skipFrames = DEFAULT_SKIP_FIRST_FRAMES;
        for (int i = 0; i < skipFrames; i++) {
            execActions("camera 0 0.0", 0);
            waitForNextObservation();
        }

        // Match main: spawn the horse after warmup skip frames so chunks are
        // loaded and the mount walk can reach it. Soft resets spawn earlier in
        // resetAgentForNewEpisode because the world is already warm.
        if (!reuseWorld) {
            mc.execute(() -> applyWorldDecorators(missionInit));
        }

        for (int i = 0; i < 5; i++) {
            execActions("camera 0 0.0", 0);
            waitForNextObservation();
        }


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
        });
    }

    private void setAgentPosition(ClientPlayerEntity player, MissionInit missionInit,
            boolean serverAuthoritative) {
        PosAndDirection startPos = getAgentStart(missionInit).getPlacement();
        if (startPos == null) {
            return;
        }
        double x = startPos.getX();
        double y = startPos.getY();
        double z = startPos.getZ();
        float yaw = startPos.getYaw();
        float pitch = startPos.getPitch();

        player.setMotion(0, 0, 0);
        player.fallDistance = 0;

        // Client-side set for an immediate local view update.
        player.setLocationAndAngles(x, y, z, yaw, pitch);
        player.rotationYaw = yaw;
        player.rotationPitch = pitch;

        if (!serverAuthoritative) {
            LOGGER.info("[Persistent] Agent reset (client only) to ({}, {}, {})", x, y, z);
            return;
        }

        Minecraft mc = Minecraft.getInstance();
        IntegratedServer integratedServer = mc.getIntegratedServer();
        if (integratedServer != null) {
            // Authoritative teleport on the server thread. Use setLocationAndAngles
            // directly — NOT connection.setPlayerLocation(), which sets targetPos
            // and makes ServerPlayNetHandler ignore client movement until
            // CConfirmTeleport, breaking the scripted mount and shifting the
            // policy start pose down-track.
            integratedServer.execute(() -> {
                ServerPlayerEntity serverPlayer =
                        integratedServer.getPlayerList().getPlayerByUUID(player.getUniqueID());
                if (serverPlayer == null) {
                    List<ServerPlayerEntity> players = integratedServer.getPlayerList().getPlayers();
                    if (!players.isEmpty()) {
                        serverPlayer = players.get(0);
                    }
                }
                if (serverPlayer != null) {
                    // Must dismount on the SERVER side first. The client-side
                    // stopRiding() earlier doesn't detach the authoritative server
                    // player, so while it's still a passenger its position is bound
                    // to the horse and the teleport gets dragged back to the vehicle.
                    if (serverPlayer.isPassenger()) {
                        serverPlayer.stopRiding();
                    }
                    serverPlayer.setMotion(0, 0, 0);
                    serverPlayer.fallDistance = 0;
                    serverPlayer.setLocationAndAngles(x, y, z, yaw, pitch);
                    serverPlayer.rotationYaw = yaw;
                    serverPlayer.rotationPitch = pitch;
                    LOGGER.info("[Persistent] Agent reset to ({}, {}, {})", x, y, z);
                } else {
                    LOGGER.warn("[Persistent] Agent teleport skipped — no server player found");
                }
            });
        } else {
            LOGGER.info("[Persistent] Agent reset (client only) to ({}, {}, {})", x, y, z);
        }
    }

    // Restore AI on the horse the agent has just mounted. The horse spawns with
    // NoAI only to hold position before the mount (otherwise its wandering AI
    // drifts it sideways off its spawn, so the agent mounts a displaced horse
    // and starts the race off the centerline). Once mounted we clear NoAI so the
    // ridden horse's server-side movement (LivingEntity.travel) matches main,
    // where the horse always has AI enabled. Deliberately does NOT teleport the
    // agent: main has no post-mount snap, and its mount walk also ends on the
    // horse (~Z=-152), so snapping to the spawn block would diverge from main's
    // actual episode-start pose.
    private void clearRiddenHorseNoAi(Minecraft mc) {
        IntegratedServer integratedServer = mc.getIntegratedServer();
        if (integratedServer == null || mc.player == null) {
            return;
        }
        final java.util.UUID playerId = mc.player.getUniqueID();
        integratedServer.execute(() -> {
            ServerPlayerEntity serverPlayer =
                    integratedServer.getPlayerList().getPlayerByUUID(playerId);
            if (serverPlayer == null) {
                List<ServerPlayerEntity> players = integratedServer.getPlayerList().getPlayers();
                if (!players.isEmpty()) {
                    serverPlayer = players.get(0);
                }
            }
            if (serverPlayer == null) {
                return;
            }
            Entity vehicle = serverPlayer.getRidingEntity();
            if (vehicle instanceof AbstractHorseEntity) {
                AbstractHorseEntity horse = (AbstractHorseEntity) vehicle;
                if (horse.isAIDisabled()) {
                    horse.setNoAI(false);
                    LOGGER.info("[HorseAI] Restored AI on mounted horse (NoAI cleared)");
                }
            }
        });
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
            LOGGER.warn("[HorseSpawn] serverHandlers is null; no decorators to apply");
            return;
        }
        int decoratorCount = serverHandlers.getWorldDecorators().size();
        LOGGER.info("[HorseSpawn] applyWorldDecorators: {} world decorator(s)", decoratorCount);
        for (Object decorator : serverHandlers.getWorldDecorators()) {
            LOGGER.info("[HorseSpawn] decorator class: {}", decorator.getClass().getName());
            if (!(decorator instanceof DrawingDecorator)) {
                continue;
            }
            DrawingDecorator drawing = (DrawingDecorator) decorator;
            int drawCount = drawing.getDrawObjectType().size();
            LOGGER.info("[HorseSpawn] DrawingDecorator has {} draw object(s)", drawCount);
            for (JAXBElement<? extends DrawObjectType> element : drawing.getDrawObjectType()) {
                LOGGER.info("[HorseSpawn] draw object: {}", element.getValue().getClass().getName());
                if (element.getValue() instanceof DrawEntity) {
                    try {
                        spawnDrawEntity((DrawEntity) element.getValue(), world, mc);
                    } catch (Exception e) {
                        LOGGER.error("[HorseSpawn] Error spawning DrawEntity", e);
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
        LOGGER.info("[HorseSpawn] spawnDrawEntity type='{}' at ({},{},{})",
            entityName, drawEntity.getX(), drawEntity.getY(), drawEntity.getZ());
        Optional<EntityType<?>> optionalType = resolveEntityType(entityName);
        if (!optionalType.isPresent()) {
            LOGGER.warn("[HorseSpawn] Unknown entity type: " + entityName);
            return;
        }

        MinecraftServer server = mc.getIntegratedServer();
        if (server == null) {
            LOGGER.warn("[HorseSpawn] Cannot spawn entity without integrated server");
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
            LOGGER.warn("[HorseSpawn] Could not spawn entity for type: " + entityName);
            return;
        }

        if (entity instanceof AbstractHorseEntity) {
            configureSpawnedHorse((AbstractHorseEntity) entity, mc, x, y, z, yaw);
        }

        serverWorld.getBlockState(spawnPos);
        if (entity instanceof AbstractHorseEntity) {
            ((AbstractHorseEntity) entity).setPositionAndUpdate(x, y, z);
        }
        LOGGER.info("[HorseSpawn] spawned {} -> isAlive={} at ({},{},{})",
            entity.getClass().getSimpleName(), entity.isAlive(), x, y, z);
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

        // Set the visual variant FIRST. readAdditional() deserializes a *full*
        // horse NBT, so every field missing from this partial tag is reset to
        // its default -- crucially setHorseTamed(getBoolean("Tame")) reads false
        // when "Tame" is absent, un-taming the horse. An untamed-but-saddled
        // horse bucks the rider (the black "angry" particles seen on dismount).
        // Doing this before the taming/saddle/attribute setup below means those
        // are applied last and stick. We also include Tame in the tag so the
        // deserialize itself keeps the horse tamed.
        if (horse instanceof HorseEntity) {
            CompoundNBT variantNbt = new CompoundNBT();
            variantNbt.putInt("Variant", HORSE_VARIANT);
            variantNbt.putBoolean("Tame", true);
            ((HorseEntity) horse).readAdditional(variantNbt);
        }

        horse.setHorseTamed(true);
        if (mc.player != null) {
            horse.setOwnerUniqueId(mc.player.getUniqueID());
        }
        horse.func_230266_a_(SoundCategory.PLAYERS);

        horse.getAttribute(Attributes.MOVEMENT_SPEED).setBaseValue(HORSE_MOVEMENT_SPEED);
        horse.getAttribute(Attributes.HORSE_JUMP_STRENGTH).setBaseValue(HORSE_JUMP_STRENGTH);
        horse.getAttribute(Attributes.MAX_HEALTH).setBaseValue(HORSE_MAX_HEALTH);
        horse.setHealth(HORSE_HEALTH);

        // Keep the horse stationary until the agent mounts it. With full AI the
        // tamed horse wanders sideways off its spawn during the mount approach,
        // so the agent mounts a displaced horse and starts the race off the
        // centerline. NoAI holds it in place; it is cleared the instant the
        // agent mounts (see clearRiddenHorseNoAi) so gameplay physics match
        // main, where the ridden horse always has AI enabled.
        horse.setNoAI(true);
        horse.setMotion(0, 0, 0);
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
        // Start with the HUD visible during the mount sequence; stepClient() hides
        // it automatically once the player is actually riding the horse (native
        // equivalent of pressing F1 after a successful mount).
        settings.hideGUI = false;

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
        long deadline = System.currentTimeMillis() + OBSERVATION_WAIT_TIMEOUT_MS;

        try {
            synchronized (pr) {
                while (envTickCounter == pr.getTickCounter()) {
                    if (System.currentTimeMillis() >= deadline) {
                        LOGGER.error(
                                "[Persistent] Timed out waiting for observation (stuck at tick {})",
                                envTickCounter);
                        envTickCounter = pr.getTickCounter();
                        return;
                    }
                    long remaining = deadline - System.currentTimeMillis();
                    pr.wait(Math.min(Math.max(remaining, 1), 50));
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
        // Hide the HUD once mounted, show it otherwise. Tied to actual riding
        // state so the GUI toggles off only after a successful horse mount.
        if (mc.player != null) {
            boolean riding = mc.player.isPassenger();
            if (mc.gameSettings != null) {
                mc.gameSettings.hideGUI = riding;
            }
            // Diagnostic: log the exact tick the player leaves the horse, and
            // whether they are still alive (alive=true => a real dismount, not a
            // death; the episode is not ending here).
            if (wasRiding && !riding) {
                LOGGER.warn("[Dismount] player left the horse at tick {} (alive={})",
                    PlayRecorder.getInstance().getTickCounter(), mc.player.isAlive());
            }
            // On mount, restore the horse's AI so ridden physics match main.
            // (No pose snap: main does not teleport post-mount either.)
            if (riding && !wasRiding) {
                clearRiddenHorseNoAi(mc);
            }
            wasRiding = riding;
        }
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
        // Add ground block & whole floor grid under the 'custom' object so MineRL's
        // Python wrapper will put them into obs['custom'].
        String groundBlockName = getGroundBlockName(mc);
        JsonArray floorGrid = buildFloorGrid(mc);

        // Add ground block and whole floor grid as top-level fields so the
        // MineRL Python wrapper exposes them directly in the raw observation.
        infoJson.addProperty("floor_block", groundBlockName);
        infoJson.add("floor_grid", floorGrid);

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
    private JsonArray buildFloorGrid(Minecraft mc) {
        JsonArray grid = new JsonArray();
        try {
            if (mc.player == null || mc.world == null) {
                return grid;
            }
            // Grid coordinate ranges (same as agent.py expects)
            int minX = -1, maxX = 1;
            int minY = -2, maxY = 0;
            int minZ = -1, maxZ = 1;

            // Malmo order: y ascending, then z ascending, then x ascending
            for (int y = minY; y <= maxY; y++) {
                for (int z = minZ; z <= maxZ; z++) {
                    for (int x = minX; x <= maxX; x++) {
                        // Player-relative coordinates: center is agent position
                        double px = mc.player.getPosX();
                        double py = mc.player.getPosY();
                        double pz = mc.player.getPosZ();

                        BlockPos pos = new BlockPos(Math.floor(px) + x, Math.floor(py) + y, Math.floor(pz) + z);
                        if (mc.world.isBlockPresent(pos)) {
                            net.minecraft.block.Block block = mc.world.getBlockState(pos).getBlock();
                            String blockName = block.toString();
                            grid.add(blockName != null && !blockName.isEmpty() ? blockName : "unknown");
                        } else {
                            grid.add("unknown");
                        }
                    }
                }
            }
        } catch (Exception e) {
            LOGGER.warn("Error building floor_grid: " + e);
        }
        return grid;
    }
    private String getGroundBlockName(Minecraft mc) {
    try {
        if (mc.player == null || mc.world == null) return "unknown";
        double px = mc.player.getPosX();
        double py = mc.player.getPosY();
        double pz = mc.player.getPosZ();
        // Two blocks down to account for player on horse (tweak HORSE offset if needed)
        BlockPos pos = new BlockPos(Math.floor(px), Math.floor(py - 2.0), Math.floor(pz));
        if (!mc.world.isBlockPresent(pos)) {
            return "unknown";
        }
        net.minecraft.block.Block block = mc.world.getBlockState(pos).getBlock();
        String blockName = block.toString();
        return blockName != null && !blockName.isEmpty() ? blockName : "unknown";
    } catch (Exception e) {
        LOGGER.warn("Error getting ground block: " + e);
        return "unknown";
    }
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
    private void quit(String command, Socket socket, boolean forceHard) throws IOException, InterruptedException {
        Minecraft mc = Minecraft.getInstance();
        if (mc.getProfiler() instanceof IResultableProfiler) {
            File profileDump = new File("profile-results-" + (new SimpleDateFormat("yyyy-MM-dd_HH.mm.ss")).format(new Date()) + ".txt");
            ((IResultableProfiler)mc.getProfiler()).getResults().writeToFile(profileDump.getAbsoluteFile());
        }
        boolean soft = !forceHard && canSoftReset();
        if (soft) {
            // Keep recording alive and unblock ReplaySender — finishAndResetEpisode()
            // stops recording and causes a game-thread / socket-thread deadlock on reset.
            ReplaySender.getInstance().clearEpisodeState();
            PlayRecorder.getInstance().softResetEpisode();
            pumpReplaySender();
            LOGGER.info("[Persistent] Soft quit — keeping integrated server and LAN clients");
        } else {
            PlayRecorder.getInstance().finishAndResetEpisode();
            if (!forceHard) {
                LOGGER.warn("[Persistent] Hard quit — soft reset unavailable (alive={} world={})",
                        integratedServerAlive, activeWorldSource);
            }
            ReplaySender.getInstance().stop();
            integratedServerAlive = false;
            activeWorldSource = null;
            lanPublished = false;
            waitForMainMenu(mc);
        }

        DataOutputStream dout = new DataOutputStream(socket.getOutputStream());
        dout.writeInt(4);
        dout.writeInt(1);
        dout.flush();
    }

    private String getConfigValue(String name, String defaultValue) {
        String prop = System.getProperty(name);
        if (prop != null && !prop.isEmpty()) {
            return prop;
        }
        String env = System.getenv(name);
        return env != null ? env : defaultValue;
    }

    private boolean runOnGameThreadWithPump(Runnable task) throws InterruptedException {
        return runOnGameThreadWithPump(task, GAME_THREAD_TASK_TIMEOUT_MS);
    }

    private boolean runOnGameThreadWithPump(Runnable task, long timeoutMs) throws InterruptedException {
        Minecraft mc = Minecraft.getInstance();
        CountDownLatch latch = new CountDownLatch(1);
        mc.execute(() -> {
            try {
                task.run();
            } finally {
                latch.countDown();
            }
        });
        long deadline = System.currentTimeMillis() + timeoutMs;
        while (latch.getCount() > 0 && System.currentTimeMillis() < deadline) {
            pumpReplaySender();
            latch.await(10, TimeUnit.MILLISECONDS);
        }
        if (latch.getCount() > 0) {
            LOGGER.error("[Persistent] Game-thread task timed out after {}ms", timeoutMs);
            return false;
        }
        return true;
    }

    private boolean runOnGameThread(BooleanSupplier check) {
        Minecraft mc = Minecraft.getInstance();
        AtomicBoolean result = new AtomicBoolean(false);
        CountDownLatch latch = new CountDownLatch(1);
        mc.execute(() -> {
            try {
                result.set(check.getAsBoolean());
            } finally {
                latch.countDown();
            }
        });
        try {
            if (!latch.await(GAME_THREAD_TIMEOUT_MS, TimeUnit.MILLISECONDS)) {
                LOGGER.warn("[Persistent] Timed out waiting for game-thread check");
                return false;
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
        return result.get();
    }

    private void logPersistentConfig(Minecraft mc) {
        if (configLogged) {
            return;
        }
        configLogged = true;
        LOGGER.info("[Persistent] Config: persistent={} lan={} port={}",
                isPersistentServerEnabled(), isLanEnabled(), getLanPort());
    }

    private void waitForPlayerReady() throws InterruptedException {
        long deadline = System.currentTimeMillis() + WORLD_READY_TIMEOUT_MS;
        int pumps = 0;
        while (System.currentTimeMillis() < deadline) {
            if (pumps++ % 5 == 0) {
                pumpReplaySender();
            }
            if (pumps % 10 == 0 && runOnGameThread(() -> {
                Minecraft mc = Minecraft.getInstance();
                return mc.world != null && mc.player != null && mc.getIntegratedServer() != null;
            })) {
                LOGGER.info("[Persistent] World and player ready");
                return;
            }
            if (pumps % 200 == 0) {
                LOGGER.info("[Persistent] Waiting for world/player to be ready...");
            }
            Thread.sleep(10);
        }
        LOGGER.error("[Persistent] Timed out waiting for world/player to be ready");
        throw new InterruptedException("World/player not ready");
    }

    private void waitForRecording() throws InterruptedException {
        PlayRecorder pr = PlayRecorder.getInstance();
        if (pr.isRecording()) {
            return;
        }
        long deadline = System.currentTimeMillis() + RECORDING_WAIT_TIMEOUT_MS;
        int pumps = 0;
        while (!pr.isRecording()) {
            // ReplaySender blocks the game thread on an empty action queue; pump
            // noop actions so PlayRecorder.tick() can call start().
            if (pumps++ % 3 == 0) {
                pumpReplaySender();
            }
            if (System.currentTimeMillis() >= deadline) {
                LOGGER.warn("[Persistent] PlayRecorder slow to start; forcing on game thread");
                forceStartRecording(pr);
                break;
            }
            if (pumps % 200 == 0) {
                LOGGER.info("[Persistent] Waiting for PlayRecorder to start...");
            }
            Thread.sleep(10);
        }
        if (!pr.isRecording()) {
            LOGGER.error("[Persistent] PlayRecorder failed to start — cannot complete mission init");
            throw new InterruptedException("PlayRecorder not recording");
        }
    }

    private void pumpReplaySender() {
        if (ReplaySender.getInstance().getMode() == ReplaySender.Mode.EXEC_CMD) {
            execActions("camera 0 0.0", 0);
        }
    }

    private void forceStartRecording(PlayRecorder pr) throws InterruptedException {
        Minecraft mc = Minecraft.getInstance();
        CountDownLatch latch = new CountDownLatch(1);
        mc.execute(() -> {
            try {
                pumpReplaySender();
                if (!pr.isRecording()) {
                    pr.start();
                }
            } catch (RuntimeException e) {
                LOGGER.error("[Persistent] PlayRecorder.start() failed", e);
            } finally {
                latch.countDown();
            }
        });
        long deadline = System.currentTimeMillis() + 15000;
        while (latch.getCount() > 0 && System.currentTimeMillis() < deadline) {
            pumpReplaySender();
            latch.await(10, TimeUnit.MILLISECONDS);
        }
        if (latch.getCount() > 0) {
            LOGGER.error("[Persistent] Timed out forcing PlayRecorder start on game thread");
        }
        if (!pr.isRecording()) {
            LOGGER.error("[Persistent] PlayRecorder still not recording after force start");
        }
    }

    private void waitForMainMenu(Minecraft mc) throws InterruptedException {
        long deadline = System.currentTimeMillis() + MAIN_MENU_WAIT_TIMEOUT_MS;
        while (true) {
            if (runOnGameThread(() -> mc.currentScreen instanceof MainMenuScreen)) {
                return;
            }
            if (System.currentTimeMillis() >= deadline) {
                LOGGER.error("[Persistent] Timed out waiting for MainMenuScreen after hard quit; proceeding");
                return;
            }
            Thread.sleep(10);
        }
    }

    private boolean isPersistentServerEnabled() {
        String enabled = getConfigValue("MINERL_PERSISTENT_SERVER", "true");
        return !enabled.equalsIgnoreCase("false");
    }

    private boolean isLanEnabled() {
        String enabled = getConfigValue("MINERL_LAN_ENABLED", "true");
        return !enabled.equalsIgnoreCase("false");
    }

    private int getLanPort() {
        int port = 25565;
        String portEnv = getConfigValue("MINERL_LAN_PORT", "25565");
        try {
            port = Integer.parseInt(portEnv);
        } catch (NumberFormatException ignored) {
        }
        return port;
    }

    private boolean canSoftReset() {
        return isPersistentServerEnabled()
                && integratedServerAlive
                && activeWorldSource != null;
    }

    private boolean canReuseWorld(String worldSrc) {
        return canSoftReset() && Objects.equals(worldSrc, activeWorldSource);
    }

    private void maybeOpenToLan() {
        if (!isLanEnabled() || lanPublished) {
            return;
        }
        int port = getLanPort();
        Minecraft mc = Minecraft.getInstance();
        mc.execute(() -> {
            IntegratedServer server = mc.getIntegratedServer();
            if (server == null) {
                LOGGER.warn("[LAN] No integrated server; skipping shareToLAN");
                return;
            }
            if (server.getPublic()) {
                lanPublished = true;
                LOGGER.info("[LAN] Already public on port {}", server.getServerPort());
                return;
            }
            boolean ok = server.shareToLAN(GameType.SPECTATOR, true, port);
            if (!ok && server.getPublic()) {
                ok = true;
            }
            if (ok) {
                lanPublished = true;
                LOGGER.info("[LAN] Spectators can connect on port {}", server.getServerPort());
            } else {
                LOGGER.warn("[LAN] shareToLAN failed on port {} (in use?)", port);
            }
        });
    }

    private void resetAgentForNewEpisode(MissionInit missionInit, boolean serverAuthoritative) {
        Minecraft mc = Minecraft.getInstance();
        if (mc.player == null) {
            LOGGER.warn("[Persistent] Cannot reset agent: player is null");
            return;
        }

        if (mc.player.isPassenger()) {
            mc.player.stopRiding();
        }
        if (!mc.player.isAlive()) {
            mc.player.respawnPlayer();
        }

        if (this.doneOnDeath) {
            mc.setHasPlayerRespawned(false);
        }

        setAgentInventory(mc.player, missionInit);
        setAgentPosition(mc.player, missionInit, serverAuthoritative);
        enforceAgentGameMode(missionInit);
        cleanupEpisodeHorses(mc);
        if (serverAuthoritative) {
            applyWorldDecorators(missionInit);
        }

        mc.player.setMotion(0, 0, 0);
        mc.player.fallDistance = 0;
    }

    private void cleanupEpisodeHorses(Minecraft mc) {
        MinecraftServer server = mc.getIntegratedServer();
        if (server == null) {
            return;
        }
        ServerWorld serverWorld = server.getWorld(World.OVERWORLD);
        // IMPORTANT: do NOT pass a world-spanning AABB here. getEntitiesWithinAABB
        // iterates chunk sections across the whole box footprint, so a +/-3e7 box
        // makes the game thread walk billions of empty sections and hang forever.
        // The race track + spawned horses live near the agent start, so bound the
        // search to a generous region centered on the player.
        double cx = mc.player.getPosX();
        double cz = mc.player.getPosZ();
        double radius = HORSE_CLEANUP_RADIUS;
        AxisAlignedBB bounds = new AxisAlignedBB(
                cx - radius, -64, cz - radius,
                cx + radius, 320, cz + radius);
        List<HorseEntity> horses = serverWorld.getEntitiesWithinAABB(EntityType.HORSE, bounds, e -> true);
        for (HorseEntity horse : horses) {
            horse.remove();
        }
        LOGGER.info("[Persistent] Removed {} horse(s) within {} blocks of agent", horses.size(), radius);
    }
}
