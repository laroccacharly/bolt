#!/usr/bin/env bun
import { mkdirSync, openSync, closeSync } from "node:fs";
import { resolve, delimiter } from "node:path";

const root = resolve(import.meta.dir, "..");
const model = resolve(root, "models/NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q4_K_XL.gguf");
const serverPath = resolve(root, "runtime/llama-b11840/llama-server");
const agentDir = resolve(root, ".local-pi");
const logPath = resolve(root, "logs/llama-server.log");
const piPath = Bun.which("pi");
if (!piPath) throw new Error("Pi is missing from PATH. Install the Pi coding harness first.");
for (const path of [model, serverPath]) {
  if (!(await Bun.file(path).exists())) throw new Error(`Missing ${path}. See README.md for setup.`);
}
for (const path of [agentDir, resolve(root, "logs")]) mkdirSync(path, { recursive: true });
const settingsPath = resolve(agentDir, "settings.json");
if (!(await Bun.file(settingsPath).exists())) {
  await Bun.write(settingsPath, JSON.stringify({
    compaction: { enabled: true, reserveTokens: 8192, keepRecentTokens: 8192 },
    branchSummary: { reserveTokens: 4096 }
  }, null, 2));
}

// Pick an available loopback port so this launch cannot attach to an unrelated model.
const reservation = Bun.listen({ hostname: "127.0.0.1", port: 0, socket: { data() {} } });
const port = reservation.port;
reservation.stop(true);
const baseUrl = `http://127.0.0.1:${port}`;
const apiKey = crypto.randomUUID();
const context = 131072;
await Bun.write(resolve(agentDir, "models.json"), JSON.stringify({
  providers: {
    "local-bolt": {
      baseUrl: `${baseUrl}/v1`, api: "openai-completions", apiKey,
      models: [{
        id: "nemotron-bolt", name: "Nemotron 3.5 Bolt E4B (local RTX 3070)",
        reasoning: true, input: ["text"], contextWindow: context, maxTokens: 8192,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        samplingParams: { temperature: 1.0, top_p: 0.95 },
        compat: {
          supportsStore: false, supportsDeveloperRole: false, supportsReasoningEffort: false,
          maxTokensField: "max_tokens", thinkingFormat: "chat-template",
          chatTemplateKwargs: { enable_thinking: { $var: "thinking.enabled" } }
        }
      }]
    }
  }
}, null, 2));

const log = openSync(logPath, "w");
const server = Bun.spawn([
  serverPath, "--model", model, "--alias", "nemotron-bolt",
  "--api-key", apiKey,
  "--host", "127.0.0.1", "--port", String(port),
  "--ctx-size", String(context), "--parallel", "1", "--n-gpu-layers", "999",
  "--jinja", "--flash-attn", "on", "--temp", "1.0", "--top-p", "0.95"
], {
  cwd: root, detached: true, stdin: "ignore", stdout: log, stderr: log,
  env: { ...process.env, LD_LIBRARY_PATH: [resolve(root, "runtime/llama-b11840"), "/opt/cuda/lib64", process.env.LD_LIBRARY_PATH].filter(Boolean).join(delimiter) }
});
closeSync(log);
let pi: ReturnType<typeof Bun.spawn> | undefined;
let stopping = false;
async function cleanup() {
  if (stopping) return;
  stopping = true;
  server.kill("SIGTERM");
  let timeout: ReturnType<typeof setTimeout> | undefined;
  await Promise.race([
    server.exited,
    new Promise<void>((resolve) => { timeout = setTimeout(resolve, 5000); })
  ]);
  clearTimeout(timeout);
  if (server.exitCode === null) server.kill("SIGKILL");
}
// Let Pi handle Ctrl+C itself; stop the model when Pi exits.
process.on("SIGINT", () => { if (!pi) { void cleanup().then(() => process.exit(130)); } });
process.on("SIGTERM", () => { pi?.kill("SIGTERM"); void cleanup().then(() => process.exit(143)); });
try {
  console.error(`Loading Bolt E4B on the GPU (${context} token context). Log: ${logPath}`);
  let ready = false;
  const deadline = Date.now() + 180_000;
  while (Date.now() < deadline) {
    if (server.exitCode !== null) throw new Error(`Model server exited (${server.exitCode}). See ${logPath}`);
    try {
      const response = await fetch(`${baseUrl}/health`, {
        headers: { Authorization: `Bearer ${apiKey}` }, signal: AbortSignal.timeout(1000)
      });
      if (response.ok) { ready = true; break; }
    } catch {}
    await Bun.sleep(500);
  }
  if (!ready) throw new Error(`Model startup timed out. See ${logPath}`);
  console.error(`Model ready at ${baseUrl}. Starting Pi; exiting Pi will release GPU memory.`);
  pi = Bun.spawn([piPath, "--offline", "--provider", "local-bolt", "--model", "nemotron-bolt", "--thinking", "off", ...process.argv.slice(2)], {
    cwd: process.cwd(), env: { ...process.env, PI_CODING_AGENT_DIR: agentDir },
    stdin: "inherit", stdout: "inherit", stderr: "inherit"
  });
  process.exitCode = await pi.exited;
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
} finally {
  await cleanup();
}
