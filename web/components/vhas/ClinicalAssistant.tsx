"use client";

import { useState, useRef, useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { CheckCircle2, HeartPulse, Loader2, AlertTriangle, ShieldCheck } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

// Resolve API URL dynamically from environment variables
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

interface WorkflowStep {
  agent_name: string;
  tool_used: string | null;
  action_summary: string;
  clinical_state_update: string;
  ui_data?: Record<string, any>;
}

interface OrchestrationResponse {
  selected_pathway: number;
  confidence_score: number;
  steps: WorkflowStep[];
}

export function ClinicalAssistant() {
  const [patientId, setPatientId] = useState("");
  const [complaint, setComplaint] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [internalThoughts, setInternalThoughts] = useState<string[]>([]);
  const [orchestrationData, setOrchestrationData] = useState<OrchestrationResponse | null>(null);
  const [isHitlTriggered, setIsHitlTriggered] = useState(false);
  const [hitlReason, setHitlReason] = useState("");
  
  const consoleEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll for the System 2 Reasoning Console
  useEffect(() => {
    consoleEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [internalThoughts]);

  const handleReset = () => {
    setPatientId("");
    setComplaint("");
    setInternalThoughts([]);
    setOrchestrationData(null);
    setIsLoading(false);
    setIsHitlTriggered(false);
    setHitlReason("");
  };

  const handleGenerate = async () => {
    if (!complaint.trim()) return;
    
    // Reset states
    setInternalThoughts([]);
    setOrchestrationData(null);
    setIsHitlTriggered(false);
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/orchestrate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nl_command: complaint }),
      });

      if (!response.body) throw new Error("No response body received from server.");
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedText = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        // Parse SSE formatted data (data: {...}\n\n)
        const lines = chunk.split("\n");
        for (const line of lines) {
            if (line.startsWith("data: ")) {
            const jsonStr = line.replace("data: ", "").trim();
            if (!jsonStr) continue;
            
            try {
              const payload = JSON.parse(jsonStr);
              if (payload.type === "thought") {
                setInternalThoughts((prev) => [...prev, payload.text]);
              } else if (payload.type === "delta") {
                accumulatedText += payload.text;
                // Attempt to parse incomplete JSON dynamically
                try {
                  const parsed = JSON.parse(accumulatedText);
                  setOrchestrationData(parsed);
                } catch {
                  // Partial JSON, ignore and wait for next chunks
                }
              } else if (payload.type === "hitl_trigger") {
                setIsHitlTriggered(true);
                setHitlReason(payload.reason || "High uncertainty detected.");
              } else if (payload.type === "done") {
                setIsLoading(false);
              }
            } catch (err) {
              console.error("Error parsing chunk:", err);
            }
          }
        }
      }
    } catch (error) {
      console.error("Inference failed:", error);
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold text-slate-900">Clinical Assistant</h2>
          <p className="text-sm text-slate-500">Intake triage context and generate validated workflows.</p>
        </div>
        <Badge variant="info" className="flex items-center gap-2 border-blue-200 bg-blue-50 text-blue-700">
          <HeartPulse className="h-4 w-4" />
          Internal Medicine - ED
        </Badge>
      </div>

      {/* INPUT INTAKE FORM */}
      {!orchestrationData && !isLoading && (
        <Card className="border-slate-200">
          <CardHeader>
            <CardTitle className="text-lg">ED Intake Form</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap items-center gap-2">
              <Badge
                onClick={() => {
                  setPatientId("ED-83012");
                  setComplaint("83-year-old female with a history of a permanent pacemaker. Presents with dizziness, weakness, and palpitations. Pacemaker interrogation notes Atrial Fibrillation.");
                }}
                className="cursor-pointer border border-blue-200 bg-blue-50 py-1.5 px-3 text-xs font-semibold text-blue-700 transition-colors hover:bg-blue-100"
              >
                Case 1: 83yo Female (AFib - System 1)
              </Badge>
              <Badge
                onClick={() => {
                  setPatientId("ED-45091");
                  setComplaint("45-year-old male presenting with sudden onset of severe, diffuse epigastric pain and nausea. History of heavy alcohol use.");
                }}
                className="cursor-pointer border border-amber-200 bg-amber-50 py-1.5 px-3 text-xs font-semibold text-amber-700 transition-colors hover:bg-amber-100"
              >
                Case 2: 45yo Male (Epigastric Pain - HITL)
              </Badge>
              <Badge
                onClick={() => {
                  setPatientId("ED-58043");
                  setComplaint("58-year-old male presenting with heavy epigastric pressure, nausea, and significant diaphoresis. History of smoking.");
                }}
                className="cursor-pointer border border-emerald-200 bg-emerald-50 py-1.5 px-3 text-xs font-semibold text-emerald-700 transition-colors hover:bg-emerald-100"
              >
                Case 3: 58yo Male (Atypical Cardiac - System 2)
              </Badge>
            </div>
            <div className="grid gap-4 md:grid-cols-[180px_1fr]">
              <div className="space-y-2">
                <label className="text-xs font-semibold uppercase tracking-wide text-slate-500">Patient ID</label>
                <Input
                  value={patientId}
                  onChange={(e) => setPatientId(e.target.value)}
                  placeholder="ED-XXXXX"
                />
              </div>
              <div className="space-y-2">
                <label className="text-xs font-semibold uppercase tracking-wide text-slate-500">Chief Complaint & History</label>
                <Textarea
                  value={complaint}
                  onChange={(e) => setComplaint(e.target.value)}
                  placeholder="Enter medical scenario context..."
                />
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-400">Decisions are decision-support only. Humans preserve liability.</span>
              <Button size="lg" onClick={handleGenerate}>
                Generate Pre-Clinical Brief
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* SYSTEM 2 REASONING CONSOLE */}
      {internalThoughts.length > 0 && (
        <Card className="border-slate-800 bg-slate-950 p-4 font-mono text-xs text-slate-200 shadow-xl">
          <CardHeader className="p-0 pb-2 border-b border-slate-800 flex flex-row items-center justify-between">
            <span className="text-emerald-400 font-bold tracking-wide">SYSTEM 2: DELIBERATIVE REASONING ENGINE</span>
            <Loader2 className="h-3 w-3 animate-spin text-emerald-400" />
          </CardHeader>
          <CardContent className="p-0 pt-3 max-h-[250px] overflow-y-auto space-y-2">
            {internalThoughts.map((thought, idx) => {
              const isError = thought.includes("❌") || thought.includes("⚠️") || thought.includes("🛑");
              const isSuccess = thought.includes("✅") || thought.includes("🟢") || thought.includes("🚀");
              return (
                <div 
                  key={idx} 
                  className={cn(
                    isError && "text-rose-400",
                    isSuccess && "text-emerald-400",
                    !isError && !isSuccess && "text-slate-300"
                  )}
                >
                  {thought}
                </div>
              );
            })}
            <div ref={consoleEndRef} />
          </CardContent>
        </Card>
      )}

      {/* HUMAN-IN-THE-LOOP INTERVENTION MODAL */}
      {isHitlTriggered && (
        <Card className="border-amber-200 bg-amber-50/70">
          <CardHeader>
            <CardTitle className="text-amber-800 flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-amber-600" />
              Human-in-the-Loop Safeguard Activated
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-sm text-amber-800">
              The dual-system detected high uncertainty. Reason: <span className="font-semibold">{hitlReason}</span>.
              Please provide clinical authorization or override to proceed.
            </p>
            <div className="flex gap-2">
              <Button variant="outline" className="border-amber-300 hover:bg-amber-100" onClick={handleReset}>
                Cancel & Re-evaluate
              </Button>
              <Button className="bg-amber-600 hover:bg-amber-700 text-white" onClick={() => setIsHitlTriggered(false)}>
                Authorize & Resume
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* GENERATED TIMELINE CARDS (DYNAMICALLY RENDERED) */}
      {orchestrationData && (
        <div className="space-y-4">
          <div className="flex justify-between items-center bg-slate-50 p-3 rounded-lg border border-slate-100">
            <span className="text-sm font-semibold">Compliance Pathway: {orchestrationData.selected_pathway}</span>
            <span className="text-sm font-semibold text-blue-600">Confidence Score: {orchestrationData.confidence_score}</span>
          </div>
          
          <div className="relative pl-8 space-y-6">
            <div className="absolute left-3 top-2 h-full w-px bg-slate-200" />
            
            {orchestrationData.steps.map((step, idx) => (
              <div key={idx} className="relative">
                <div className="absolute -left-8 top-1.5 flex h-6 w-6 items-center justify-center rounded-full bg-emerald-100 border border-emerald-300 text-emerald-700">
                  <CheckCircle2 className="h-4 w-4" />
                </div>
                <Card className="border-slate-200">
                  <CardHeader className="py-3">
                    <CardTitle className="text-sm font-semibold text-slate-800">{step.agent_name}</CardTitle>
                    {step.tool_used && <span className="text-xs text-slate-400">Tool: {step.tool_used}</span>}
                  </CardHeader>
                  <CardContent className="text-sm text-slate-600 space-y-2">
                    <p>{step.action_summary}</p>
                    <p className="text-xs italic text-slate-400">State Update: {step.clinical_state_update}</p>
                  </CardContent>
                </Card>
              </div>
            ))}
          </div>

          <div className="flex justify-end pt-4">
            <Button size="lg" onClick={handleReset}>Start New Intake</Button>
          </div>
        </div>
      )}
    </div>
  );
}