"use client";

import { useState, useRef, useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { CheckCircle2, HeartPulse, Loader2, AlertTriangle, ShieldCheck, FileText, ClipboardList, Activity, Pill, UserCheck } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";

const cn = (...classes: any[]) => classes.filter(Boolean).join(" ");

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

interface WorkflowStep {
  agent_name: string;
  tool_used: string | null;
  action_summary: string;
  clinical_state_update: string;
  ui_data?: Record<string, any>;
}

const extractCompletedSteps = (text: string): { steps: WorkflowStep[], parsedLength: number } => {
  const stepsStart = text.indexOf('"steps"');
  if (stepsStart === -1) return { steps: [], parsedLength: 0 };
  
  const arrayStart = text.indexOf('[', stepsStart);
  if (arrayStart === -1) return { steps: [], parsedLength: 0 };
  
  let searchIndex = arrayStart + 1;
  const parsedSteps: WorkflowStep[] = [];
  
  while (true) {
    const nextBrace = text.indexOf('{', searchIndex);
    if (nextBrace === -1) break;
    
    let braceCount = 1;
    let index = nextBrace + 1;
    
    while (braceCount > 0 && index < text.length) {
      const char = text[index];
      if (char === '{') braceCount++;
      else if (char === '}') braceCount--;
      index++;
    }
    
    if (braceCount === 0) {
      const stepStr = text.substring(nextBrace, index);
      try {
        const stepObj = JSON.parse(stepStr);
        parsedSteps.push(stepObj);
        searchIndex = index; 
      } catch {
        break; 
      }
    } else {
      break; 
    }
  }
  
  return { steps: parsedSteps, parsedLength: searchIndex };
};

export function ClinicalAssistant() {
  const [patientId, setPatientId] = useState("");
  const [complaint, setComplaint] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [internalThoughts, setInternalThoughts] = useState<string[]>([]);
  const [isHitlTriggered, setIsHitlTriggered] = useState(false);
  const [hitlReason, setHitlReason] = useState("");
  
  const [selectedPathway, setSelectedPathway] = useState<number | null>(null);
  const [confidenceScore, setConfidenceScore] = useState<number | null>(null);
  const [steps, setSteps] = useState<WorkflowStep[]>([]);
  
  const consoleEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    consoleEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [internalThoughts]);

  const handleReset = () => {
    setPatientId("");
    setComplaint("");
    setInternalThoughts([]);
    setSteps([]);
    setSelectedPathway(null);
    setConfidenceScore(null);
    setIsLoading(false);
    setIsHitlTriggered(false);
    setHitlReason("");
  };

  const handleGenerate = async () => {
    if (!complaint.trim()) return;
    
    setInternalThoughts([]);
    setSteps([]);
    setSelectedPathway(null);
    setConfidenceScore(null);
    setIsHitlTriggered(false);
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/orchestrate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nl_command: complaint }),
      });

      if (!response.body) throw new Error("No response body received.");
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedText = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
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
                
                // 1. Extract Pathway and Confidence dynamically from raw text using Regex
                if (!selectedPathway) {
                  const pathwayMatch = accumulatedText.match(/"selected_pathway"\s*:\s*(\d)/);
                  if (pathwayMatch) setSelectedPathway(parseInt(pathwayMatch[1]));
                }
                if (!confidenceScore) {
                  const confidenceMatch = accumulatedText.match(/"confidence_score"\s*:\s*([\d.]+)/);
                  if (confidenceMatch) setConfidenceScore(parseFloat(confidenceMatch[1]));
                }

                // 2. Extract and update completed steps in real-time
                const { steps: parsedSteps } = extractCompletedSteps(accumulatedText);
                if (parsedSteps.length > steps.length) {
                  setSteps(parsedSteps);
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

  const getAcuityColor = (level?: string) => {
    switch (level?.toUpperCase()) {
      case "HIGH": return "bg-rose-100 text-rose-700 border-rose-200";
      case "MEDIUM": return "bg-amber-100 text-amber-700 border-amber-200";
      case "LOW": return "bg-emerald-100 text-emerald-700 border-emerald-200";
      default: return "bg-slate-100 text-slate-700 border-slate-200";
    }
  };

  const renderAgentCard = (step: WorkflowStep) => {
    const data = step.ui_data || {};
    
    switch (step.agent_name) {
      case "TriageAgent":
        return (
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <span className="text-xs font-semibold text-slate-500">Acuity Level:</span>
              <Badge variant="default" className={getAcuityColor(data.acuity_level)}>
                {data.acuity_level || "UNKNOWN"}
              </Badge>
            </div>
            {data.suspected_condition && (
              <p className="text-sm">
                <span className="font-bold text-slate-800">Suspected Condition: </span>
                {data.suspected_condition}
              </p>
            )}
            <p className="text-sm text-slate-600 bg-slate-50 p-3 rounded border border-slate-100">
              {data.rationale || step.action_summary}
            </p>
          </div>
        );

      case "EHRAgent":
        return (
          <div className="space-y-4">
            {data.vitals && (
              <div className="grid grid-cols-2 gap-2 md:grid-cols-4 animate-fade-in">
                {Object.entries(data.vitals).map(([key, value]) => (
                  <div key={key} className="bg-slate-50 p-2 rounded border border-slate-100 text-center">
                    <span className="text-[10px] uppercase tracking-wider font-semibold text-slate-400">{key}</span>
                    <p className="text-sm font-bold text-slate-700">{value as string}</p>
                  </div>
                ))}
              </div>
            )}
            {data.critical_alert && (
              <div className="bg-rose-50 border border-rose-200 text-rose-800 p-3 rounded text-xs font-semibold flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 text-rose-600 animate-pulse" />
                {data.critical_alert}
              </div>
            )}
            {data.ehr_note && <p className="text-sm text-slate-600">{data.ehr_note}</p>}
          </div>
        );

      case "DispensationAgent":
        return (
          <div className="bg-blue-50/50 border border-blue-100 p-3 rounded space-y-2">
            <p className="text-sm"><span className="font-bold text-slate-800">Action Taken: </span>{data.action_taken || step.action_summary}</p>
            {data.clinical_goal && <p className="text-xs italic text-slate-500">Goal: {data.clinical_goal}</p>}
          </div>
        );

      case "ReconciliationAgent":
        return (
          <div className="bg-purple-50/50 border border-purple-100 p-3 rounded space-y-2">
            <p className="text-sm"><span className="font-bold text-slate-800">Safety Check: </span>{data.action_taken || step.action_summary}</p>
            {data.clinical_goal && <p className="text-xs italic text-slate-500">Objective: {data.clinical_goal}</p>}
          </div>
        );

      case "SummaryAgent":
        return (
          <div className="grid gap-4 md:grid-cols-2 pt-2">
            <div className="space-y-2 md:col-span-2">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Handover Summary</span>
              <p className="text-sm text-slate-700 bg-slate-50 p-3 rounded border border-slate-100">{data.summary_text || step.action_summary}</p>
            </div>
            {data.extracted_history && (
              <div className="space-y-1">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Extracted History</span>
                <ul className="list-disc pl-4 text-xs text-slate-600 space-y-1">
                  {data.extracted_history.map((h: string, i: number) => <li key={i}>{h}</li>)}
                </ul>
              </div>
            )}
            {data.ai_proposed_next_steps && (
              <div className="space-y-1">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">AI Proposed Next Steps</span>
                <ul className="list-disc pl-4 text-xs text-slate-600 space-y-1">
                  {data.ai_proposed_next_steps.map((s: string, i: number) => <li key={i}>{s}</li>)}
                </ul>
              </div>
            )}
          </div>
        );

      default:
        return <p className="text-sm text-slate-600">{step.action_summary}</p>;
    }
  };

  const getAgentBlockTitle = (name: string) => {
    switch (name) {
      case "TriageAgent": return "Triage Assessment";
      case "EHRAgent": return "Initial Vitals & EHR Retrieval";
      case "DispensationAgent": return "Emergency Dispensation";
      case "ReconciliationAgent": return "Medication Reconciliation";
      case "SummaryAgent": return "Final Handover Brief";
      default: return name;
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

      {steps.length === 0 && !isLoading && (
        <Card className="border-slate-200 shadow-sm animate-fade-in">
          <CardHeader>
            <CardTitle className="text-base font-semibold">ED Intake Form</CardTitle>
          </CardHeader>
          <CardContent className="space-y-5">
            <div className="space-y-2">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-400">Quick-Click Scenario Examples:</span>
              <div className="flex flex-wrap gap-2">
                <Badge
                  variant="default"
                  className="cursor-pointer hover:bg-slate-100 transition-colors py-1.5 px-3 border-slate-200 text-xs font-medium text-slate-700"
                  onClick={() => {
                    setPatientId("ED-83012");
                    setComplaint("83-year-old female with a history of a permanent pacemaker. Presents with dizziness, weakness, and palpitations. Pacemaker interrogation notes Atrial Fibrillation.");
                  }}
                >
                  🟢 Case 1: 83yo Female (AFib - System 1)
                </Badge>
                <Badge
                  variant="default"
                  className="cursor-pointer hover:bg-slate-100 transition-colors py-1.5 px-3 border-slate-200 text-xs font-medium text-slate-700"
                  onClick={() => {
                    setPatientId("ED-45091");
                    setComplaint("45-year-old male presenting with sudden onset of severe, diffuse epigastric pain and nausea. History of heavy alcohol use.");
                  }}
                >
                  🟡 Case 2: 45yo Male (Epigastric - HITL)
                </Badge>
                <Badge
                  variant="default"
                  className="cursor-pointer hover:bg-slate-100 transition-colors py-1.5 px-3 border-slate-200 text-xs font-medium text-slate-700"
                  onClick={() => {
                    setPatientId("ED-58043");
                    setComplaint("58-year-old male presenting with heavy epigastric pressure, nausea, and significant diaphoresis. History of smoking.");
                  }}
                >
                  🔵 Case 3: 58yo Male (Atypical Cardiac - System 2)
                </Badge>
              </div>
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
                  className="min-h-[100px]"
                />
              </div>
            </div>
            <div className="flex items-center justify-between pt-2 border-t border-slate-100">
              <span className="text-xs text-slate-400">Decisions are decision-support only. Humans preserve liability.</span>
              <Button size="lg" onClick={handleGenerate}>
                Generate Pre-Clinical Brief
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {internalThoughts.length > 0 && (
        <Card className="border-slate-800 bg-slate-950 p-4 font-mono text-xs text-slate-200 shadow-xl animate-fade-in">
          <CardHeader className="p-0 pb-2 border-b border-slate-800 flex flex-row items-center justify-between">
            <span className="text-emerald-400 font-bold tracking-wide">SYSTEM 2: DELIBERATIVE REASONING ENGINE</span>
            <Loader2 className="h-3 w-3 animate-spin text-emerald-400" />
          </CardHeader>
          <CardContent className="p-0 pt-3 max-h-[300px] overflow-y-auto space-y-2">
            {internalThoughts.map((thought, idx) => {
              const isError = thought.includes("❌") || thought.includes("⚠️") || thought.includes("🛑");
              const isSuccess = thought.includes("✅") || thought.includes("🟢") || thought.includes("🚀");
              return (
                <div 
                  key={idx} 
                  className={
                    isError ? "text-rose-400" : isSuccess ? "text-emerald-400" : "text-slate-300"
                  }
                >
                  {thought}
                </div>
              );
            })}
            <div ref={consoleEndRef} />
          </CardContent>
        </Card>
      )}

      {isHitlTriggered && (
        <Card className="border-amber-200 bg-amber-50/70 animate-fade-in">
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

      {selectedPathway && !isHitlTriggered && (
        <div className="space-y-4 animate-fade-in">
          <div className="flex justify-between items-center bg-slate-50 px-4 py-3 rounded-lg border border-slate-100">
            <span className="text-sm font-semibold text-slate-700">Compliance Pathway: {selectedPathway}</span>
            <span className="text-sm font-semibold text-blue-600">
              Confidence Score: {confidenceScore !== null ? confidenceScore.toFixed(2) : "Calculating..."}
            </span>
          </div>
          
          <div className="relative pl-8 space-y-6">
            <div className="absolute left-3 top-2 h-[calc(100%-24px)] w-px bg-slate-200" />
            
            <AnimatePresence>
              {steps.map((step, idx) => (
                <motion.div 
                  key={idx} 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4 }}
                  className="relative"
                >
                  <div className="absolute -left-8 top-1.5 flex h-6 w-6 items-center justify-center rounded-full bg-emerald-100 border border-emerald-300 text-emerald-700">
                    <CheckCircle2 className="h-4 w-4" />
                  </div>
                  <Card className="border-slate-200 shadow-sm">
                    <CardHeader className="py-3 border-b border-slate-50 flex flex-row items-center justify-between">
                      <CardTitle className="text-sm font-bold text-slate-800">
                        {getAgentBlockTitle(step.agent_name)}
                      </CardTitle>
                      {step.tool_used && (
                        <Badge variant="default" className="text-[10px] font-mono px-2 py-0.5">
                          Tool: {step.tool_used}
                        </Badge>
                      )}
                    </CardHeader>
                    <CardContent className="pt-3 pb-4 space-y-2">
                      {renderAgentCard(step)}
                      <p className="text-[10px] italic text-slate-400 border-t border-slate-50 pt-2">
                        State Update: {step.clinical_state_update}
                      </p>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </AnimatePresence>

            {isLoading && (
              <div className="relative animate-pulse">
                <div className="absolute -left-8 top-1.5 flex h-6 w-6 items-center justify-center rounded-full bg-slate-100 border border-slate-200 text-slate-400">
                  <Loader2 className="h-4 w-4 animate-spin" />
                </div>
                <Card className="border-slate-100 bg-slate-50/50">
                  <div className="p-4 space-y-3">
                    <div className="h-4 bg-slate-200 rounded w-1/4" />
                    <div className="space-y-2">
                      <div className="h-3 bg-slate-200 rounded w-3/4" />
                      <div className="h-3 bg-slate-200 rounded w-1/2" />
                    </div>
                  </div>
                </Card>
              </div>
            )}
          </div>

          {!isLoading && (
            <div className="flex justify-end pt-4">
              <Button size="lg" onClick={handleReset}>Start New Intake</Button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}