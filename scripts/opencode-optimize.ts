import { optimizeConfig } from "../src/opencode-config.ts";

const arguments_ = new Set(process.argv.slice(2));
const unknown = [...arguments_].filter((argument) => argument !== "--dry-run");
if (unknown.length > 0) {
  console.error(`Unknown argument: ${unknown.join(", ")}`);
  process.exitCode = 1;
} else {
  try {
    const result = await optimizeConfig(undefined, { dryRun: arguments_.has("--dry-run") });
    console.log("OpenCode token optimization\n");
    console.log(`Config: ${result.configPath}\n`);
    if (result.changes.length === 0) {
      console.log("Already optimized; no changes made.");
    } else {
      console.log(result.dryRun ? "Would update:" : "Updated:");
      for (const change of result.changes) {
        const oldValue = change.before === undefined ? "missing" : JSON.stringify(change.before);
        const newValue = change.after === undefined ? "missing" : JSON.stringify(change.after);
        console.log(`  ${change.path}: ${oldValue} -> ${newValue}`);
      }
      if (result.dryRun) {
        console.log("\nDry run: no files or backup were written.");
      } else if (result.backupCreated) {
        console.log(`\nBackup:\n  ${result.backupPath}`);
      } else if (result.before !== undefined) {
        console.log(`\nBackup retained:\n  ${result.backupPath}`);
      }
    }
  } catch (error) {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  }
}
