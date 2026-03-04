"use client";

import { useMemo, useState } from "react";
import { Eye, Network, ShieldCheck, Workflow } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import { useI18n } from "@/components/vhas/i18n";
import otlpDemoTraces from "@/data/otlp_demo_traces.json";

const traces = [
  {
    id: "TR-2024-1101",
    patient: "ED-66712",
    status: "Complete",
    confidence: 0.9,
    otlpTraceId: "a1c4e9b8d6f1405c9e0a3f9b2c8d771a",
    steps: [
      {
        title: "Triage Assessment",
        titleVi: "Đánh giá Cấp cứu",
        reasoning:
          "Suspected Condition: Lung Fibrosis Exacerbation. Rationale: Based on progressive dyspnea and dry cough.",
        reasoningVi:
          "Chẩn đoán nghi ngờ: Đợt cấp xơ phổi. Lý do: Khó thở tăng dần và ho khan.",
      },
      {
        title: "Vitals & EHR Retrieval",
        titleVi: "Sinh hiệu & EHR",
        reasoning:
          "CRITICAL ALERT: SpO2 89% (Hypoxia). Patient is showing signs of severe respiratory distress.",
        reasoningVi:
          "CẢNH BÁO: SpO2 89% (Thiếu oxy). Bệnh nhân có dấu hiệu suy hô hấp nặng.",
      },
      {
        title: "Final Handover Brief",
        titleVi: "Bàn giao Lâm sàng",
        reasoning:
          "66-year-old male with lung fibrosis presents with worsening dyspnea. Triaged as high priority. Vitals notable for hypoxia with O2 saturation of 89%. Ready for physician evaluation for supplemental oxygen and consideration for admission.",
        reasoningVi:
          "Nam 66 tuổi xơ phổi, khó thở nặng hơn. Ưu tiên cao. Sinh hiệu cho thấy thiếu oxy SpO2 89%. Sẵn sàng đánh giá và cân nhắc nhập viện.",
      },
    ],
  },
  {
    id: "TR-2024-1102",
    patient: "ED-8392",
    status: "Complete",
    confidence: 0.92,
    otlpTraceId: "c6d8e1f2a3b4c5d6e7f8091a2b3c4d5e",
    steps: [
      {
        title: "Triage Assessment",
        titleVi: "Đánh giá Cấp cứu",
        reasoning:
          "Suspected Condition: Atrial Fibrillation with Rapid Ventricular Response (AFib w/ RVR). Rationale: Based on palpitations, weakness, and pacemaker history.",
        reasoningVi:
          "Chẩn đoán nghi ngờ: Rung nhĩ có đáp thất nhanh (AFib RVR). Lý do: Hồi hộp, yếu, tiền sử máy tạo nhịp.",
      },
      {
        title: "Initial Vitals & EHR Retrieval",
        titleVi: "Sinh hiệu ban đầu & EHR",
        reasoning:
          "CRITICAL ALERT: Patient is hemodynamically unstable (Hypotension + Severe Tachycardia).",
        reasoningVi:
          "CẢNH BÁO: Huyết động không ổn định (HA thấp + Nhịp nhanh).",
      },
      {
        title: "Emergency Dispensation",
        titleVi: "Can thiệp Cấp cứu",
        reasoning:
          "Action Taken: Administered 10mg Diltiazem IV push. Clinical Goal: Urgent heart rate control.",
        reasoningVi:
          "Hành động: Tiêm tĩnh mạch 10mg Diltiazem. Mục tiêu: Kiểm soát nhịp tim khẩn cấp.",
      },
      {
        title: "Post-Intervention Vitals Re-check",
        titleVi: "Sinh hiệu sau can thiệp",
        reasoning:
          "Hemodynamics improved. Heart rate successfully controlled post-IV Diltiazem.",
        reasoningVi:
          "Huyết động cải thiện. Nhịp tim đã được kiểm soát sau Diltiazem.",
      },
      {
        title: "Medication Reconciliation",
        titleVi: "Đối chiếu Thuốc",
        reasoning:
          "WARNING: Patient Non-Compliance Detected. EHR and pharmacy records indicate the patient has been out of their prescribed anticoagulant (Apixaban) for 3 days. High risk for thromboembolism.",
        reasoningVi:
          "CẢNH BÁO: Không tuân thủ thuốc. EHR và nhà thuốc cho thấy bệnh nhân ngừng Apixaban 3 ngày. Nguy cơ huyết khối cao.",
      },
      {
        title: "Final Handover Brief",
        titleVi: "Bàn giao Lâm sàng",
        reasoning:
          "83yo female with pacemaker presented with AFib with RVR and hypotension. Heart rate now controlled post-IV Diltiazem. Med recon reveals 3-day non-compliance with Apixaban. Patient stabilized, ready for admission and Cardiology consult.",
        reasoningVi:
          "Nữ 83 tuổi có máy tạo nhịp, AFib RVR và hạ huyết áp. Nhịp tim đã được kiểm soát sau Diltiazem. Đối chiếu thuốc: bỏ Apixaban 3 ngày. Bệnh nhân ổn định, sẵn sàng nhập viện và hội chẩn Tim mạch.",
      },
    ],
  },
  {
    id: "TR-2024-1103",
    patient: "ED-5092",
    status: "Review",
    confidence: 0.45,
    otlpTraceId: "5f9d1c2b3a4e6f708192a3b4c5d6e7f8",
    steps: [
      {
        title: "Triage Assessment",
        titleVi: "Đánh giá Cấp cứu",
        reasoning:
          "Suspected Condition: Undifferentiated Severe Epigastric Pain. Rationale: Patient has severe upper abdominal pain with a history of alcohol use. Need to differentiate between Acute Pancreatitis and Severe Peptic Ulcer Bleeding.",
        reasoningVi:
          "Chẩn đoán nghi ngờ: Đau thượng vị nặng chưa rõ nguyên nhân. Lý do: Đau bụng trên nặng và tiền sử rượu. Cần phân biệt viêm tụy cấp và xuất huyết loét dạ dày.",
      },
      {
        title: "Vitals & EHR Retrieval",
        titleVi: "Sinh hiệu & EHR",
        reasoning:
          "EHR Note: No prior endoscopies on file. No known allergies.",
        reasoningVi:
          "Ghi chú EHR: Không có nội soi trước đó. Không ghi nhận dị ứng.",
      },
      {
        title: "HITL Confirmation",
        titleVi: "Xác nhận lâm sàng",
        reasoning:
          "Confidence Score: 45%. Critical clinical data is missing to determine the correct Internal Medicine pathway. Please confirm the primary physical/history finding.",
        reasoningVi:
          "Độ tin cậy: 45%. Thiếu thông tin quan trọng để chọn đúng đường nội khoa. Vui lòng xác nhận dấu hiệu chính.",
      },
      {
        title: "Final Handover Brief",
        titleVi: "Bàn giao Lâm sàng",
        reasoning:
          "45yo male with severe epigastric pain and tachycardia. Clinician confirms pain radiating to the back with intractable vomiting. Highly suspicious for Acute Pancreatitis (Likely alcohol-induced).",
        reasoningVi:
          "Nam 45 tuổi đau thượng vị nặng, nhịp tim nhanh. Bác sĩ xác nhận đau lan ra sau lưng và nôn không kiểm soát. Nghi ngờ cao viêm tụy cấp (liên quan rượu).",
      },
    ],
  },
];

export function TraceExplorer() {
  const { locale } = useI18n();
  const isVi = locale === "vi";
  const [activeTrace, setActiveTrace] = useState(traces[0]);
  const [expandedStep, setExpandedStep] = useState("step-0");
  const otlpTraceById = useMemo(() => {
    const map = new Map<string, (typeof otlpDemoTraces)[number]>();
    otlpDemoTraces.forEach((trace) => {
      map.set(trace.trace_id, trace);
    });
    return map;
  }, []);

  const otlpJson = useMemo(() => {
    const otlpTrace = otlpTraceById.get(activeTrace.otlpTraceId);
    return JSON.stringify(otlpTrace ?? {}, null, 2);
  }, [activeTrace.otlpTraceId, otlpTraceById]);

  const highlightedJson = useMemo(() => {
    const tokens: React.ReactNode[] = [];
    const json = otlpJson;
    const regex = /("(?:\\.|[^"\\])*")(?=\s*:)|("(?:\\.|[^"\\])*")|\b(true|false|null)\b|-?\d+(?:\.\d+)?/g;
    let lastIndex = 0;
    let match: RegExpExecArray | null;
    let tokenIndex = 0;

    while ((match = regex.exec(json)) !== null) {
      if (match.index > lastIndex) {
        tokens.push(json.slice(lastIndex, match.index));
      }

      const [full, keyString, valueString, boolNull] = match;

      if (keyString) {
        tokens.push(
          <span key={`k-${tokenIndex++}`} className="text-sky-700">
            {keyString}
          </span>
        );
      } else if (valueString) {
        tokens.push(
          <span key={`s-${tokenIndex++}`} className="text-emerald-700">
            {valueString}
          </span>
        );
      } else if (boolNull) {
        tokens.push(
          <span key={`b-${tokenIndex++}`} className="text-rose-600">
            {boolNull}
          </span>
        );
      } else {
        tokens.push(
          <span key={`n-${tokenIndex++}`} className="text-amber-600">
            {full}
          </span>
        );
      }

      lastIndex = match.index + full.length;
    }

    if (lastIndex < json.length) {
      tokens.push(json.slice(lastIndex));
    }

    return tokens;
  }, [otlpJson]);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold text-slate-900">
          {isVi ? "Truy vết Quyết định" : "Trace Explorer"}
        </h2>
        <p className="text-sm text-slate-500">
          {isVi
            ? "Kiểm toán chuỗi lý do lâm sàng với minh bạch theo lớp."
            : "Audit the clinical reasoning sequence with layered transparency."}
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
        <Card className="h-full">
          <CardHeader>
            <CardTitle>{isVi ? "Truy vết gần đây" : "Recent Traces"}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {traces.map((trace) => (
              <button
                key={trace.id}
                onClick={() => {
                  setActiveTrace(trace);
                  setExpandedStep("step-0");
                }}
                className={cn(
                  "flex w-full items-start justify-between rounded-lg border px-3 py-3 text-left transition",
                  activeTrace.id === trace.id
                    ? "border-blue-200 bg-blue-50"
                    : "border-slate-200 hover:border-slate-300"
                )}
              >
                <div>
                  <p className="text-sm font-semibold text-slate-900">{trace.id}</p>
                  <p className="text-xs text-slate-500">
                    {isVi ? "Bệnh nhân" : "Patient"} {trace.patient}
                  </p>
                  <p className="mt-1 text-xs font-medium text-slate-500">
                    {isVi ? "Độ tin cậy" : "Confidence"}: {Math.round(trace.confidence * 100)}%
                  </p>
                </div>
                <Badge variant={trace.status === "Review" ? "warning" : "success"}>
                  {trace.status === "Review"
                    ? isVi
                      ? "Cần xem"
                      : "Review"
                    : isVi
                    ? "Hoàn thành"
                    : "Complete"}
                </Badge>
              </button>
            ))}
          </CardContent>
        </Card>

        <div className="space-y-4">
          <Card>
            <CardHeader className="space-y-2">
              <div className="flex items-center justify-between">
                <CardTitle>{isVi ? "Chi tiết truy vết" : "Trace Detail"}</CardTitle>
                <div className="flex items-center gap-2">
                  <Badge variant="info">{activeTrace.id}</Badge>
                  <Badge variant="default">
                    {isVi ? "Độ tin cậy" : "Confidence"}: {Math.round(activeTrace.confidence * 100)}%
                  </Badge>
                </div>
              </div>
              <p className="text-sm text-slate-500">
                {isVi
                  ? "Kiến trúc lớp: bước lâm sàng, lớp lý do, dữ liệu OTLP."
                  : "Onion architecture: clinical steps, reasoning layer, raw OTLP data."}
              </p>
            </CardHeader>
            <CardContent className="space-y-4">
              <Accordion
                type="single"
                collapsible
                value={expandedStep}
                onValueChange={(value) => setExpandedStep(value || "step-0")}
                className="space-y-3"
              >
                {activeTrace.steps.map((step, index) => {
                  const value = `step-${index}`;
                  return (
                    <AccordionItem key={step.title} value={value}>
                      <AccordionTrigger>
                        <div className="flex items-center gap-3">
                          <span
                            className={cn(
                              "flex h-9 w-9 items-center justify-center rounded-lg",
                              expandedStep === value ? "bg-blue-100" : "bg-slate-100"
                            )}
                          >
                            {index === 0 ? (
                              <Workflow className="h-5 w-5 text-blue-700" />
                            ) : index === 1 ? (
                              <Network className="h-5 w-5 text-blue-700" />
                            ) : (
                              <ShieldCheck className="h-5 w-5 text-blue-700" />
                            )}
                          </span>
                          <div>
                            <p className="text-sm font-semibold text-slate-900">
                              {isVi ? step.titleVi : step.title}
                            </p>
                            <p className="text-xs text-slate-500">
                              {isVi ? "Bước lâm sàng" : "Clinical step"}
                            </p>
                          </div>
                        </div>
                      </AccordionTrigger>
                      <AccordionContent>
                        <div className="mt-2 rounded-md border border-slate-200 bg-slate-100 px-3 py-2 font-mono text-xs text-slate-700">
                          <p className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">
                            {isVi ? "Giải thích lâm sàng" : "Layer 2: Clinical reasoning"}
                          </p>
                          <p className="mt-2 leading-relaxed">
                            {isVi ? step.reasoningVi : step.reasoning}
                          </p>
                        </div>
                      </AccordionContent>
                    </AccordionItem>
                  );
                })}
              </Accordion>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>{isVi ? "Dữ liệu OTLP" : "Raw OTLP Data"}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Dialog>
                <DialogTrigger asChild>
                  <Button variant="outline" className="flex items-center gap-2">
                    <Eye className="h-4 w-4" />
                    {isVi ? "Xem OTLP JSON" : "View OTLP JSON"}
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>{isVi ? "Gói OTLP" : "OTLP Trace Payload"}</DialogTitle>
                    <DialogDescription>
                      {isVi
                        ? `Dữ liệu kiểm toán cho ${activeTrace.id}.`
                        : `Structured audit payload for ${activeTrace.id}.`}
                    </DialogDescription>
                  </DialogHeader>
                  <pre className="mt-4 max-h-[60vh] overflow-auto rounded-xl border border-slate-200 bg-slate-50 p-4 text-xs leading-relaxed">
                    <code className="font-mono whitespace-pre-wrap">
                      {highlightedJson}
                    </code>
                  </pre>
                </DialogContent>
              </Dialog>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
