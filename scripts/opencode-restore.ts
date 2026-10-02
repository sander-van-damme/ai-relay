import { defaultConfigPaths, restoreConfig } from "../src/opencode-config.ts";

const paths = defaultConfigPaths();
try {
  await restoreConfig(paths);
  console.log("OpenCode token optimization restore\n");
  console.log(`Restored: ${paths.configPath}`);
  console.log(`Backup:   ${paths.backupPath}`);
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
}
