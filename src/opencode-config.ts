import { constants } from "node:fs";
import { copyFile, mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import ts from "typescript";

export const OPTIMIZATIONS = [
  { path: ["warming"], value: false },
  { path: ["compaction", "auto"], value: true },
  { path: ["compaction", "keep", "tokens"], value: 3_000 },
  { path: ["compaction", "buffer"], value: 4_000 },
  { path: ["tool_output", "max_lines"], value: 1_000 },
  { path: ["tool_output", "max_bytes"], value: 32_768 },
  { path: ["agents", "title", "disabled"], value: true },
] as const;

const RELAY_AUTO_MODEL_OPTIMIZATIONS = [
  { path: ["provider", "relay", "models", "auto", "limit", "context"], value: 18_000 },
  { path: ["provider", "relay", "models", "auto", "limit", "input"], value: 18_000 },
  { path: ["provider", "relay", "models", "auto", "limit", "output"], value: 4_000 },
] as const;

export interface ConfigPaths {
  configPath: string;
  backupPath: string;
}

export interface Change {
  path: string;
  before: unknown;
  after: boolean | number | string | undefined;
}

export interface OptimizeResult extends ConfigPaths {
  changes: Change[];
  backupCreated: boolean;
  configCreated: boolean;
  dryRun: boolean;
  before: string | undefined;
  after: string;
}

export function defaultConfigPaths(): ConfigPaths {
  const configPath = join(homedir(), ".config", "opencode", "opencode.jsonc");
  return { configPath, backupPath: `${configPath}.backup` };
}

function parse(text: string): ts.JsonSourceFile {
  const source = ts.parseJsonText("opencode.jsonc", text);
  const diagnostics = (source as ts.JsonSourceFile & { parseDiagnostics: readonly ts.Diagnostic[] }).parseDiagnostics;
  if (diagnostics.length > 0) {
    const diagnostic = diagnostics[0]!;
    const location = source.getLineAndCharacterOfPosition(diagnostic.start ?? 0);
    throw new Error(
      `Cannot optimize invalid JSONC at ${location.line + 1}:${location.character + 1}: ${ts.flattenDiagnosticMessageText(diagnostic.messageText, "\n")}`,
    );
  }
  if (!source.statements[0] || !ts.isExpressionStatement(source.statements[0]) ||
      !ts.isObjectLiteralExpression(source.statements[0].expression)) {
    throw new Error("OpenCode configuration must contain a JSONC object at its root");
  }
  return source;
}

function rootObject(source: ts.JsonSourceFile): ts.ObjectLiteralExpression {
  return (source.statements[0] as ts.ExpressionStatement).expression as ts.ObjectLiteralExpression;
}

function propertyNamed(object: ts.ObjectLiteralExpression, name: string): ts.PropertyAssignment | undefined {
  return object.properties.find(
    (property): property is ts.PropertyAssignment =>
      ts.isPropertyAssignment(property) &&
      ((ts.isStringLiteral(property.name) && property.name.text === name) ||
        (ts.isIdentifier(property.name) && property.name.text === name)),
  );
}

function decodedValue(node: ts.Expression): unknown {
  if (node.kind === ts.SyntaxKind.TrueKeyword) return true;
  if (node.kind === ts.SyntaxKind.FalseKeyword) return false;
  if (ts.isNumericLiteral(node)) return Number(node.text);
  if (ts.isStringLiteral(node)) return node.text;
  if (node.kind === ts.SyntaxKind.NullKeyword) return null;
  return "[non-scalar value]";
}

function lineIndent(text: string, position: number): string {
  const lineStart = text.lastIndexOf("\n", position - 1) + 1;
  return text.slice(lineStart, position).match(/^\s*/)?.[0] ?? "";
}

function childIndent(text: string, source: ts.JsonSourceFile, object: ts.ObjectLiteralExpression): string {
  const first = object.properties[0];
  if (first) return lineIndent(text, first.getStart(source));
  return `${lineIndent(text, object.getStart(source))}  `;
}

function nestedObject(path: readonly string[], value: boolean | number | string, indent: string): string {
  if (path.length === 0) return JSON.stringify(value);
  const [head, ...tail] = path;
  return `{\n${indent}  ${JSON.stringify(head)}: ${nestedObject(tail, value, `${indent}  `)}\n${indent}}`;
}

function addProperty(
  text: string,
  source: ts.JsonSourceFile,
  object: ts.ObjectLiteralExpression,
  name: string,
  valueText: string,
): string {
  const close = object.getEnd() - 1;
  const indent = childIndent(text, source, object);
  const closingIndent = lineIndent(text, object.getStart(source));
  const last = object.properties.at(-1);
  if (last) {
    let insertion = last.getEnd();
    if (object.properties.hasTrailingComma) {
      insertion = text.indexOf(",", insertion) + 1;
    } else {
      text = `${text.slice(0, insertion)},${text.slice(insertion)}`;
      insertion += 1;
    }
    const adjustedClose = close + (object.properties.hasTrailingComma ? 0 : 1);
    const trailing = text.slice(insertion, adjustedClose);
    // Keep end-of-property comments where they are rather than inserting into them.
    if (/\/\*|\/\//.test(trailing)) {
      insertion = adjustedClose;
    }
    return `${text.slice(0, insertion)}\n${indent}${JSON.stringify(name)}: ${valueText}${text.slice(insertion)}`;
  }
  const trivia = text.slice(object.getStart(source) + 1, close);
  const prefix = trivia.endsWith("\n") ? "" : "\n";
  return `${text.slice(0, close)}${prefix}${indent}${JSON.stringify(name)}: ${valueText}\n${closingIndent}${text.slice(close)}`;
}

function setPath(
  text: string,
  path: readonly string[],
  value: boolean | number | string,
): { text: string; before: unknown; changed: boolean } {
  const source = parse(text);
  let object = rootObject(source);

  for (let index = 0; index < path.length; index += 1) {
    const name = path[index]!;
    const property = propertyNamed(object, name);
    const isLeaf = index === path.length - 1;
    if (!property) {
      const indent = childIndent(text, source, object);
      const valueText = nestedObject(path.slice(index + 1), value, indent);
      return { text: addProperty(text, source, object, name, valueText), before: undefined, changed: true };
    }
    if (isLeaf) {
      const before = decodedValue(property.initializer);
      if (before === value) return { text, before, changed: false };
      return {
        text: `${text.slice(0, property.initializer.getStart(source))}${JSON.stringify(value)}${text.slice(property.initializer.getEnd())}`,
        before,
        changed: true,
      };
    }
    if (!ts.isObjectLiteralExpression(property.initializer)) {
      const before = decodedValue(property.initializer);
      const indent = lineIndent(text, property.getStart(source));
      const replacement = nestedObject(path.slice(index + 1), value, indent);
      return {
        text: `${text.slice(0, property.initializer.getStart(source))}${replacement}${text.slice(property.initializer.getEnd())}`,
        before,
        changed: true,
      };
    }
    object = property.initializer;
  }
  return { text, before: undefined, changed: false };
}


function removePath(
  text: string,
  path: readonly string[],
  expectedValue?: boolean | number | string,
): { text: string; before: unknown; changed: boolean } {
  const source = parse(text);
  let object = rootObject(source);

  for (let index = 0; index < path.length; index += 1) {
    const name = path[index]!;
    const property = propertyNamed(object, name);
    if (!property) return { text, before: undefined, changed: false };

    const isLeaf = index === path.length - 1;
    if (!isLeaf) {
      if (!ts.isObjectLiteralExpression(property.initializer)) {
        return { text, before: undefined, changed: false };
      }
      object = property.initializer;
      continue;
    }

    const before = decodedValue(property.initializer);
    if (expectedValue !== undefined && before !== expectedValue) {
      return { text, before, changed: false };
    }

    const properties = [...object.properties];
    const propertyIndex = properties.indexOf(property);
    let start = property.getStart(source);
    let end = property.getEnd();
    const next = properties[propertyIndex + 1];
    const previous = properties[propertyIndex - 1];
    const objectClose = object.getEnd() - 1;

    if (next) {
      const comma = text.indexOf(",", end);
      if (comma >= 0 && comma < next.getStart(source)) end = comma + 1;
    } else if (object.properties.hasTrailingComma) {
      const comma = text.indexOf(",", end);
      if (comma >= 0 && comma < objectClose) end = comma + 1;
    } else if (previous) {
      const comma = text.lastIndexOf(",", start);
      if (comma >= previous.getEnd()) start = comma;
    }

    return {
      text: `${text.slice(0, start)}${text.slice(end)}`,
      before,
      changed: true,
    };
  }

  return { text, before: undefined, changed: false };
}

function hasObjectPath(text: string, path: readonly string[]): boolean {
  const source = parse(text);
  let object = rootObject(source);
  for (const [index, name] of path.entries()) {
    const property = propertyNamed(object, name);
    if (!property) return false;
    if (index === path.length - 1) return ts.isObjectLiteralExpression(property.initializer);
    if (!ts.isObjectLiteralExpression(property.initializer)) return false;
    object = property.initializer;
  }
  return false;
}

export function applyOptimizations(input: string): { text: string; changes: Change[] } {
  let text = input;
  const changes: Change[] = [];
  parse(text);

  // Older optimizer versions disabled OpenCode's skill tool globally. Remove
  // only that legacy value so Paperclip and other runtimes can load skills on
  // demand, while preserving any explicit non-deny user preference.
  const staleSkillDeny = removePath(text, ["permission", "skill"], "deny");
  text = staleSkillDeny.text;
  if (staleSkillDeny.changed) {
    changes.push({ path: "permission.skill", before: staleSkillDeny.before, after: undefined });
  }

  const optimizeRelayAuto = hasObjectPath(text, ["provider", "relay", "models", "auto"]);
  const optimizations = optimizeRelayAuto
    ? [...OPTIMIZATIONS, ...RELAY_AUTO_MODEL_OPTIMIZATIONS]
    : OPTIMIZATIONS;
  for (const optimization of optimizations) {
    const result = setPath(text, optimization.path, optimization.value);
    text = result.text;
    if (result.changed) {
      changes.push({ path: optimization.path.join("."), before: result.before, after: optimization.value });
    }
  }
  parse(text);
  return { text, changes };
}

async function readOptional(path: string): Promise<string | undefined> {
  try {
    return await readFile(path, "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
    throw error;
  }
}

async function atomicWrite(path: string, contents: string): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.tmp-${process.pid}-${Date.now()}`;
  try {
    await writeFile(temporary, contents, { encoding: "utf8", mode: 0o600 });
    await rename(temporary, path);
  } finally {
    await rm(temporary, { force: true });
  }
}

export async function optimizeConfig(
  paths: ConfigPaths = defaultConfigPaths(),
  options: { dryRun?: boolean } = {},
): Promise<OptimizeResult> {
  const before = await readOptional(paths.configPath);
  const initial = before === undefined || before.trim() === "" ? "{}\n" : before;
  const applied = applyOptimizations(initial);
  const dryRun = options.dryRun ?? false;
  let backupCreated = false;

  if (!dryRun && applied.changes.length > 0) {
    await mkdir(dirname(paths.configPath), { recursive: true });
    if (before !== undefined) {
      try {
        await copyFile(paths.configPath, paths.backupPath, constants.COPYFILE_EXCL);
        backupCreated = true;
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
      }
    }
    await atomicWrite(paths.configPath, applied.text);
  }

  return {
    ...paths,
    changes: applied.changes,
    backupCreated,
    configCreated: before === undefined && applied.changes.length > 0,
    dryRun,
    before,
    after: applied.text,
  };
}

export async function restoreConfig(paths: ConfigPaths = defaultConfigPaths()): Promise<void> {
  const backup = await readOptional(paths.backupPath);
  if (backup === undefined) {
    throw new Error(`No OpenCode configuration backup found at ${paths.backupPath}`);
  }
  await atomicWrite(paths.configPath, backup);
}
