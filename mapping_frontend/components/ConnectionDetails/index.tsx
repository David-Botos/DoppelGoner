// components/ConnectionDetails/index.tsx
import React from "react";
import { VisualizationEntityEdge, EntityDetails } from "../../lib/types";
import ActionButtons from "./ActionButtons";
import EntityDetailsCard from "./EntityDetailsCard";
import MatchEvidencePanel from "./EnhancedMatchEvidencePanel";

interface ConnectionDetailsProps {
  connection: VisualizationEntityEdge | null;
  entityDetailsCache: Record<string, EntityDetails>;
  onConfirm: () => void;
  onReject: () => void;
  submitting: boolean;
  error: Error | null;
}

const ConnectionDetails: React.FC<ConnectionDetailsProps> = ({
  connection,
  entityDetailsCache,
  onConfirm,
  onReject,
  submitting,
  error,
}) => {
  if (!connection) {
    return (
      <div className="p-6 text-center text-gray-500 h-full flex items-center justify-center">
        <p>
          Select a connection in the graph to review.
        </p>
      </div>
    );
  }

  const entity1Details = entityDetailsCache[connection.entity_id_1];
  const entity2Details = entityDetailsCache[connection.entity_id_2];

  return (
    <div className="p-4 h-full overflow-auto bg-gray-50">
      <div className="mb-3">
        <h2 className="text-xl font-bold">Entity Link Review</h2>
        <p className="text-gray-600 text-sm">
          Evaluate whether these entities represent the same real-world organization
        </p>
      </div>

      {/* Match Evidence */}
      <div className="mb-4">
        <h3 className="font-medium text-lg mb-2 flex items-center">
          <svg className="w-5 h-5 mr-1 text-gray-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          Match Evidence
        </h3>
        <div className="bg-white p-4 rounded-lg shadow-sm">
          <MatchEvidencePanel connection={connection} />
        </div>
      </div>

      {/* Decision Section */}
      <div className="mb-4">
        <h3 className="font-medium text-lg mb-2">Decision</h3>
        <div className="bg-white p-4 rounded-lg shadow-sm">
          <p className="text-sm text-gray-700 mb-3">
            Do these two entities represent the same real-world organization?
          </p>
          <ActionButtons
            onConfirm={onConfirm}
            onReject={onReject}
            submitting={submitting}
          />
          
          {error && (
            <div className="mt-3 p-2 bg-red-100 text-red-800 rounded text-sm">
              Error: {error.message}
            </div>
          )}
        </div>
      </div>

      {/* Entity Details Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <h3 className="font-medium text-lg mb-2">Entity 1</h3>
          <EntityDetailsCard
            entityId={connection.entity_id_1}
            details={entity1Details}
            loading={!entity1Details}
          />
        </div>
        
        <div>
          <h3 className="font-medium text-lg mb-2">Entity 2</h3>
          <EntityDetailsCard
            entityId={connection.entity_id_2}
            details={entity2Details}
            loading={!entity2Details}
          />
        </div>
      </div>
    </div>
  );
};

export default ConnectionDetails;