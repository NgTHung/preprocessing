// Adapt the upstream executable plugin to MyST's JavaScript plugin interface.
// Node cannot spawn Python scripts directly on Windows; use the uv environment's
// installed executable instead. Keep all directive and transform logic upstream.
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const executable = fileURLToPath(new URL(
  process.platform === "win32"
    ? "../../.venv/Scripts/jupyter-book-marimo.exe"
    : "../../.venv/bin/jupyter-book-marimo",
  import.meta.url,
));

function invoke(args = [], data) {
  return JSON.parse(execFileSync(executable, args, {
    encoding: "utf8",
    input: data === undefined ? undefined : JSON.stringify(data),
    maxBuffer: 32 * 1024 * 1024,
  }));
}

const spec = invoke();

export default {
  ...spec,
  directives: (spec.directives ?? []).map((directive) => ({
    ...directive,
    run: (data) => invoke(["--directive", directive.name], data),
  })),
  transforms: (spec.transforms ?? []).map((transform) => ({
    ...transform,
    plugin: () => async (tree) => {
      Object.assign(tree, invoke(["--transform", transform.name], tree));
    },
  })),
};
