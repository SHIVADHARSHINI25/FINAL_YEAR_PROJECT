import { useEffect, useState } from 'react';
import { ReactFlow, Background, MarkerType, type Node, type Edge } from '@xyflow/react';
import '@xyflow/react/dist/style.css';

const initialNodes: Node[] = [
  { id: 'orch', position: { x: 250, y: 0 }, data: { label: 'Orchestrator' }, type: 'input' },
  { id: 'fda', position: { x: 50, y: 100 }, data: { label: 'Failure Detection (FDA)' } },
  { id: 'rca', position: { x: 50, y: 200 }, data: { label: 'Root Cause Analysis' } },
  { id: 'dqa', position: { x: 200, y: 150 }, data: { label: 'Data Quality' } },
  { id: 'maa', position: { x: 350, y: 150 }, data: { label: 'Model Analysis' } },
  { id: 'rma', position: { x: 500, y: 150 }, data: { label: 'Resource Monitor' } },
  { id: 'de', position: { x: 250, y: 280 }, data: { label: 'Decision Engine' } },
  { id: 'kra', position: { x: 500, y: 280 }, data: { label: 'Knowledge Base' } },
  { id: 'va', position: { x: 250, y: 380 }, data: { label: 'Verification Agent' }, type: 'output' },
];

const initialEdges: Edge[] = [
  { id: 'e-orch-fda', source: 'orch', target: 'fda', markerEnd: { type: MarkerType.ArrowClosed } },
  { id: 'e-fda-rca', source: 'fda', target: 'rca', markerEnd: { type: MarkerType.ArrowClosed } },
  { id: 'e-rca-dqa', source: 'rca', target: 'dqa', markerEnd: { type: MarkerType.ArrowClosed } },
  { id: 'e-rca-maa', source: 'rca', target: 'maa', markerEnd: { type: MarkerType.ArrowClosed } },
  { id: 'e-rca-rma', source: 'rca', target: 'rma', markerEnd: { type: MarkerType.ArrowClosed } },
  { id: 'e-dqa-de', source: 'dqa', target: 'de', markerEnd: { type: MarkerType.ArrowClosed } },
  { id: 'e-maa-de', source: 'maa', target: 'de', markerEnd: { type: MarkerType.ArrowClosed } },
  { id: 'e-rma-de', source: 'rma', target: 'de', markerEnd: { type: MarkerType.ArrowClosed } },
  { id: 'e-de-kra', source: 'de', target: 'kra', animated: true, style: { strokeDasharray: '5,5' } },
  { id: 'e-de-va', source: 'de', target: 'va', markerEnd: { type: MarkerType.ArrowClosed } },
];

export default function AgentNetwork({ activeEvent }: { activeEvent: string }) {
  const [nodes, setNodes] = useState<Node[]>(initialNodes);
  const [edges, setEdges] = useState<Edge[]>(initialEdges);

  useEffect(() => {
    let activeNodes: string[] = ['orch'];
    let animatedEdges: string[] = [];

    if (activeEvent === 'failure_detected') {
      activeNodes = ['orch', 'fda'];
      animatedEdges = ['e-orch-fda'];
    } else if (activeEvent === 'rca_completed') {
      activeNodes = ['fda', 'rca'];
      animatedEdges = ['e-fda-rca'];
    } else if (activeEvent === 'deep_dive_completed') {
      activeNodes = ['rca', 'dqa', 'maa', 'rma'];
      animatedEdges = ['e-rca-dqa', 'e-rca-maa', 'e-rca-rma'];
    } else if (activeEvent === 'decision_engine_ranked') {
      activeNodes = ['dqa', 'maa', 'rma', 'de', 'kra'];
      animatedEdges = ['e-dqa-de', 'e-maa-de', 'e-rma-de', 'e-de-kra'];
    } else if (activeEvent === 'verification_completed' || activeEvent === 'run_completed') {
      activeNodes = ['de', 'va', 'orch'];
      animatedEdges = ['e-de-va'];
    }

    setNodes(nds => nds.map(node => ({
      ...node,
      style: {
        background: activeNodes.includes(node.id) ? '#4f46e5' : '#1e293b',
        color: 'white',
        border: activeNodes.includes(node.id) ? '2px solid #818cf8' : '1px solid #334155',
        borderRadius: '8px',
        padding: '10px',
        boxShadow: activeNodes.includes(node.id) ? '0 0 15px rgba(79, 70, 229, 0.6)' : 'none',
        transition: 'all 0.3s ease',
      }
    })));

    setEdges(eds => eds.map(edge => ({
      ...edge,
      animated: animatedEdges.includes(edge.id) || (edge.id === 'e-de-kra' && activeEvent === 'decision_engine_ranked'),
      style: { stroke: animatedEdges.includes(edge.id) ? '#818cf8' : '#334155', strokeWidth: animatedEdges.includes(edge.id) ? 2 : 1 }
    })));

  }, [activeEvent]);

  return (
    <div style={{ height: '350px', width: '100%' }}>
      <ReactFlow nodes={nodes} edges={edges} fitView colorMode="dark">
        <Background color="#334155" gap={16} />
      </ReactFlow>
    </div>
  );
}
