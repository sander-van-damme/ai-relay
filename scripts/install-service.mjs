import { execFileSync } from "node:child_process";
import {
  chmod,
  copyFile,
  cp,
  mkdir,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const installDir = "/opt/ai-relay";
const envTarget = "/etc/ai-relay.env";
const serviceTarget = "/etc/systemd/system/ai-relay.service";

function run(command, args, options = {}) {
  execFileSync(command, args, { stdio: "inherit", ...options });
}

function runBestEffort(command, args, options = {}) {
  try {
    run(command, args, options);
  } catch {
    // Diagnostics should not replace the original installation error.
  }
}

function requireRoot() {
  if (typeof process.getuid === "function" && process.getuid() !== 0) {
    throw new Error("Run this command as root: sudo npm run install-service");
  }
}

function requireNodeVersion() {
  const [major, minor] = process.versions.node.split(".").map(Number);
  if ((major ?? 0) < 24 || ((major ?? 0) === 24 && (minor ?? 0) < 12)) {
    throw new Error(`Node.js 24.12+ is required; found ${process.versions.node}`);
  }
}

function ensureServiceUser() {
  try {
    execFileSync("id", ["-u", "ai-relay"], { stdio: "ignore" });
  } catch {
    run("useradd", ["--system", "--home", installDir, "--shell", "/usr/sbin/nologin", "ai-relay"]);
  }
}

async function installFiles() {
  await mkdir(installDir, { recursive: true });

  for (const entry of ["src", "config"]) {
    await rm(resolve(installDir, entry), { recursive: true, force: true });
    await cp(resolve(repoRoot, entry), resolve(installDir, entry), { recursive: true });
  }
  await copyFile(resolve(repoRoot, "package.json"), resolve(installDir, "package.json"));

  const localEnv = resolve(repoRoot, ".env");
  if (existsSync(localEnv)) {
    await copyFile(localEnv, envTarget);
  } else if (!existsSync(envTarget)) {
    await copyFile(resolve(repoRoot, ".env.example"), envTarget);
  }
  await chmod(envTarget, 0o600);

  const serviceTemplate = await readFile(resolve(repoRoot, "deploy/ai-relay.service"), "utf8");
  const service = serviceTemplate.replace(
    /^ExecStart=.*$/m,
    `ExecStart=${process.execPath} ${installDir}/src/index.ts`,
  );
  await writeFile(serviceTarget, service, "utf8");

}

function installProductionDependencies() {
  const args = ["install", "--omit=dev", "--no-audit", "--no-fund"];
  const npmExecPath = process.env.npm_execpath;

  if (npmExecPath) {
    run(process.execPath, [npmExecPath, ...args], { cwd: installDir });
  } else {
    run("npm", args, { cwd: installDir });
  }

  run("chown", ["-R", "ai-relay:ai-relay", installDir]);
}

async function readPort() {
  const config = JSON.parse(await readFile(resolve(repoRoot, "config/relay.json"), "utf8"));
  return Number(config?.server?.port ?? 8787);
}

function printServiceDiagnostics() {
  console.error("\nAI Relay failed its health check. Service diagnostics:");
  runBestEffort("systemctl", ["status", "ai-relay", "--no-pager", "-l"]);
  console.error("\nRecent AI Relay logs:");
  runBestEffort("journalctl", ["-u", "ai-relay", "-n", "50", "--no-pager"]);
}

async function verifyHealth() {
  const port = await readPort();
  const healthUrl = `http://127.0.0.1:${port}/health`;
  let lastError;
  for (let attempt = 0; attempt < 20; attempt += 1) {
    try {
      const response = await fetch(healthUrl);
      if (response.ok) return;
      lastError = new Error(`health endpoint returned HTTP ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 250));
  }
  throw lastError instanceof Error ? lastError : new Error("health check failed");
}

async function main() {
  requireRoot();
  requireNodeVersion();
  run("systemctl", ["--version"], { stdio: "ignore" });
  ensureServiceUser();
  await installFiles();
  installProductionDependencies();

  run("systemctl", ["daemon-reload"]);
  run("systemctl", ["enable", "ai-relay"]);
  run("systemctl", ["restart", "ai-relay"]);
  try {
    await verifyHealth();
  } catch (error) {
    printServiceDiagnostics();
    throw error;
  }

  const port = await readPort();
  console.log(`AI Relay is installed and running at http://127.0.0.1:${port}`);
  console.log(`Provider secrets: ${envTarget}`);
  console.log("Logs: journalctl -u ai-relay -f");
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
