import { getMetrics } from "../lib/api";
import { useAsync, usePageTitle } from "../lib/hooks";
import { fmtMs, fmtNum, fmtPct } from "../lib/format";
import { BarList } from "../components/charts";
import { Card, EmptyState, ErrorState, LoadingState, PageHeader } from "../components/ui";

/**
 * GET /api/metrics returns training/reports/metrics.json, filled step by step by the training
 * scripts. Any section can be missing (that step has not been run yet), so every section
 * renders a "not run yet" state on its own.
 */
export default function MetricsPage() {
  usePageTitle("Model metrics");
  const m = useAsync((signal) => getMetrics({ signal }), []);

  return (
    <>
      <PageHeader
        title="Model metrics"
        description="Offline evaluation written by training/evaluate.py and training/export_onnx.py. mAP50-95 is COCO's main metric (averaged over IoU 0.50…0.95)."
      />
      {m.loading && !m.data ? (
        <LoadingState />
      ) : m.error ? (
        <ErrorState message={m.error} onRetry={m.reload} />
      ) : (
        <div className="space-y-4">
          <CocoBaseline data={m.data?.coco_baseline} />
          <DomainGap data={m.data?.domain_gap} />
          <Finetuned data={m.data?.finetuned} />
          <ExportedModels data={m.data?.models} />
        </div>
      )}
    </>
  );
}

const isEmpty = (r) => !r || Object.keys(r).length === 0;

function NotRun({ cmd }) {
  return (
    <EmptyState>
      Not run yet. Generate it with <code className="rounded bg-surface-2 px-1.5 py-0.5 text-xs">{cmd}</code>
    </EmptyState>
  );
}

function Table({ head, rows }) {
  return (
    <div className="-mx-4 overflow-x-auto sm:mx-0">
      <table className="table min-w-[32rem]">
        <thead>
          <tr>
            {head.map((h, i) => (
              <th key={i} scope="col" className={i === 0 ? "" : "text-right"}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {r.map((c, j) => (
                <td key={j} className={j === 0 ? "font-medium" : "text-right tabular-nums"}>
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CocoBaseline({ data }) {
  return (
    <Card title="1 · COCO baseline (pretrained models, 500 val images)">
      {isEmpty(data) ? (
        <NotRun cmd="python training/evaluate.py coco-baseline" />
      ) : (
        <Table
          head={["Model", "mAP50-95", "mAP50", "AP small", "AP medium", "AP large", "Latency", "Params"]}
          rows={Object.entries(data).map(([name, r]) => [
            name,
            fmtNum(r.mAP50_95),
            fmtNum(r.mAP50),
            fmtNum(r.AP_small),
            fmtNum(r.AP_medium),
            fmtNum(r.AP_large),
            fmtMs(r.latency_ms),
            r.params_M !== undefined ? `${r.params_M.toFixed(1)} M` : "—",
          ])}
        />
      )}
    </Card>
  );
}

function DomainGap({ data }) {
  return (
    <Card title="2 · Domain gap (COCO-pretrained models on VisDrone val, shared classes)">
      {isEmpty(data) ? (
        <NotRun cmd="python training/evaluate.py domain-gap" />
      ) : (
        <>
          <p className="mb-3 text-sm text-muted">
            How well a model trained on everyday photos works on drone footage <em>before</em> fine-tuning — compare
            with the fine-tuned rows below.
          </p>
          <Table
            head={["Setting", "mAP50-95", "mAP50", "AP small", "AP medium", "AP large"]}
            rows={Object.entries(data).map(([name, r]) => [
              name,
              fmtNum(r.mAP50_95),
              fmtNum(r.mAP50),
              fmtNum(r.AP_small),
              fmtNum(r.AP_medium),
              fmtNum(r.AP_large),
            ])}
          />
        </>
      )}
    </Card>
  );
}

function sizeRow(label, e) {
  return [label, fmtNum(e?.mAP50_95), fmtNum(e?.mAP50), fmtNum(e?.AP_small), fmtNum(e?.AP_medium), fmtNum(e?.AP_large)];
}

function Finetuned({ data }) {
  return (
    <Card title="3 · Fine-tuned on VisDrone">
      {isEmpty(data) ? (
        <NotRun cmd="python training/evaluate.py finetuned" />
      ) : (
        <div className="space-y-6">
          <Table
            head={["Model", "mAP50-95", "mAP50", "Precision", "Recall"]}
            rows={Object.entries(data).map(([name, r]) => [
              name,
              fmtNum(r.mAP50_95),
              fmtNum(r.mAP50),
              fmtPct(r.precision),
              fmtPct(r.recall),
            ])}
          />
          <div className="grid gap-6 lg:grid-cols-2">
            {Object.entries(data).map(([name, r]) => (
              <FinetunedModel key={name} name={name} r={r} />
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}

function FinetunedModel({ name, r }) {
  const perClass = Object.entries(r.per_class_AP50_95 ?? {}).sort((x, y) => y[1] - x[1]);
  return (
    <div className="min-w-0 space-y-3 rounded-md border border-border p-3">
      <h3 className="font-medium">{name}</h3>
      <div>
        <div className="label">Per-class AP50-95</div>
        <BarList data={perClass} format={(v) => v.toFixed(3)} max={1} emptyText="No per-class AP recorded." />
      </div>
      <div>
        <div className="label">By object size (pycocotools)</div>
        {r.by_size || r.shared_classes_for_domain_gap ? (
          <Table
            head={["Subset", "mAP50-95", "mAP50", "AP S", "AP M", "AP L"]}
            rows={[
              ...(r.by_size ? [sizeRow("All classes", r.by_size)] : []),
              ...(r.shared_classes_for_domain_gap ? [sizeRow("Shared w/ COCO", r.shared_classes_for_domain_gap)] : []),
            ]}
          />
        ) : (
          <p className="text-sm text-muted">Not recorded.</p>
        )}
      </div>
    </div>
  );
}

function parityCell(p) {
  if (!p) return "—";
  return (
    <span
      title={`${p.onnx_boxes} ONNX boxes vs ${p.reference_boxes} PyTorch boxes; max score diff ${p.max_score_diff ?? "—"}`}
    >
      {fmtPct(p.match_rate)} · IoU {fmtNum(p.mean_iou_of_matches)}
    </span>
  );
}

function ExportedModels({ data }) {
  return (
    <Card title="4 · Exported ONNX models (served by the API)">
      {isEmpty(data) ? (
        <NotRun cmd="python training/export_onnx.py" />
      ) : (
        <>
          <p className="mb-3 text-sm text-muted">
            Parity = share of PyTorch boxes reproduced by our ONNX pipeline with IoU &gt; 0.9 (checks
            pre/post-processing). Latency is per image; “—” means that device was not available.
          </p>
          <Table
            head={[
              "Model",
              "ONNX size",
              "Parity (FP32)",
              "Parity (INT8)",
              "CPU FP32",
              "CPU INT8",
              "CUDA FP32",
              "CUDA FP16",
            ]}
            rows={Object.entries(data).map(([name, r]) => [
              name,
              r.onnx_mb !== undefined ? `${r.onnx_mb.toFixed(1)} MB` : "—",
              parityCell(r.parity),
              parityCell(r.int8_parity),
              fmtMs(r.latency_ms?.cpu_fp32),
              fmtMs(r.latency_ms?.cpu_int8),
              fmtMs(r.latency_ms?.cuda_fp32),
              fmtMs(r.latency_ms?.cuda_fp16),
            ])}
          />
        </>
      )}
    </Card>
  );
}
