// Official Khronos glTF-Validator runner (same engine as https://github.khronos.org/glTF-Validator/).
// Usage: node tools/gltf-validator/validate.mjs <file.gltf|file.glb> [--report out.json] [--max-issues N]
// Exit code: 0 = no errors and no warnings, 1 = warnings present, 2 = errors present, 3 = usage / IO failure.
import { readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import validator from "gltf-validator";

const args = process.argv.slice(2);
if (args.length === 0) {
  console.error("Usage: node validate.mjs <asset.gltf> [--report out.json] [--max-issues N]");
  process.exit(3);
}
const assetPath = path.resolve(args[0]);
const reportIdx = args.indexOf("--report");
const reportPath = reportIdx >= 0 ? path.resolve(args[reportIdx + 1]) : null;
const maxIdx = args.indexOf("--max-issues");
const maxIssues = maxIdx >= 0 ? parseInt(args[maxIdx + 1], 10) : 100000;

const assetDir = path.dirname(assetPath);
const bytes = new Uint8Array(await readFile(assetPath));

const report = await validator.validateBytes(bytes, {
  uri: path.basename(assetPath),
  maxIssues,
  // Load external resources (e.g. creature_diffuse.png) relative to the asset so image references are validated too.
  externalResourceFunction: (uri) =>
    readFile(path.resolve(assetDir, decodeURIComponent(uri))).then((b) => new Uint8Array(b)),
});

const issues = report.issues;
const counts = {
  errors: issues.numErrors,
  warnings: issues.numWarnings,
  infos: issues.numInfos,
  hints: issues.numHints,
};

const byCode = {};
for (const m of issues.messages) {
  byCode[m.code] = (byCode[m.code] || 0) + 1;
}

console.log(`Khronos glTF-Validator ${report.validatorVersion}`);
console.log(`Asset     : ${assetPath}`);
console.log(`Errors    : ${counts.errors}`);
console.log(`Warnings  : ${counts.warnings}`);
console.log(`Infos     : ${counts.infos}`);
console.log(`Hints     : ${counts.hints}`);
if (report.info) {
  const i = report.info;
  console.log(
    `Stats     : drawCalls=${i.drawCallCount} animations=${i.animationCount} materials=${i.materialCount} ` +
      `vertices=${i.totalVertexCount} triangles=${i.totalTriangleCount} maxInfluences=${i.maxInfluences}`
  );
  if (i.resources) {
    for (const r of i.resources) {
      console.log(`Resource  : ${r.pointer} ${r.storage} ${r.mimeType || ""} ${r.uri || ""}`.trim());
    }
  }
}
for (const [code, n] of Object.entries(byCode)) {
  console.log(`  ${code}: ${n}`);
}

if (reportPath) {
  await mkdir(path.dirname(reportPath), { recursive: true });
  await writeFile(reportPath, JSON.stringify(report, null, 2));
  console.log(`Report    : ${reportPath}`);
}

// Use exitCode (not process.exit) so piped stdout is fully flushed on Windows.
process.exitCode = counts.errors > 0 ? 2 : counts.warnings > 0 ? 1 : 0;
